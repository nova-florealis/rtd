import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoPipelineForText2Image, DPMSolverSinglestepScheduler, EulerDiscreteScheduler
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers.models import UNet2DConditionModel
from diffusers import AutoencoderTiny

from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig


def get_diffusion_dimensions(height_diffusion_desired, width_diffusion_desired, latent_div=16, autoenc_div=8):
    """
    Calculate appropriate dimensions for diffusion that are divisible by latent_div and autoenc_div
    """
    height_latents = round(height_diffusion_desired / autoenc_div)
    height_latents = round(latent_div * height_latents / latent_div)
    height_diffusion_corrected = int(height_latents * autoenc_div)

    width_latents = round(width_diffusion_desired / autoenc_div)
    width_latents = round(latent_div * width_latents / latent_div)
    width_diffusion_corrected = int(width_latents * autoenc_div)

    if height_diffusion_corrected != height_diffusion_desired or width_diffusion_corrected != width_diffusion_desired:
        print(f"Autocorrected the desired dimensions. Corrected: ({height_diffusion_corrected}, {width_diffusion_corrected}), Desired: ({height_diffusion_desired}, {width_diffusion_desired})")

    return height_diffusion_corrected, width_diffusion_corrected, height_latents, width_latents


class SimpleDiffusionEngine:
    """
    A minimal version of the DiffusionEngine class that provides only core functionality
    using the standard diffusers library
    """
    def __init__(
        self,
        height_diffusion_desired=512,
        width_diffusion_desired=512,
        use_image2image=True,
        use_tinyautoenc=True,
        device='cuda:0',
        hf_model = "stabilityai/stable-diffusion-xl-base-1.0", #'stabilityai/sdxl-turbo',
        repo = "ByteDance/SDXL-Lightning",
        # ckpt = "sdxl_lightning_2step_unet.safetensors",
        do_compile=False,
        do_diffusion=True,
        use_lightning=False,
        use_loras=False,
        lora_configs=None,
        lora_scale=1.0, # Default LoRA scale
    ):
        self._init_resolution(height_diffusion_desired, width_diffusion_desired)
        self.do_compile = do_compile
        self.do_diffusion = do_diffusion
        self.use_tinyautoenc = use_tinyautoenc
        self.device = device
        self.hf_model = hf_model

        self.num_inference_steps = 4 # 2. 4. 6. 8 literal

        self.repo = repo
        self.ckpt = f"sdxl_lightning_{self.num_inference_steps}step_unet.safetensors"
        
        self.seed = 420
        self.guidance_scale = 0.0
        self.strength = 0.5
        self.latents = None
        self.embeds = None
        self.image_init = None
        self.modulations = {}
        
        # LoRA configuration
        self.use_loras = use_loras
        self.lora_scale = lora_scale

        # Load LoRA configurations if provided
        self.lora_configs = lora_configs
        
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False 

        self.use_image2image = use_image2image
        self.use_lightning = use_lightning

        if not self.do_diffusion:
            self.device = 'cpu'
            self.do_compile = False

        if self.use_image2image:
            self._init_image2image()
        else:
            self._init_text2image()

    def _init_resolution(self, height_diffusion_desired, width_diffusion_desired):
        height_diffusion_corrected, width_diffusion_corrected, height_latents, width_latents = get_diffusion_dimensions(height_diffusion_desired, width_diffusion_desired)
        self.height_latents = height_latents
        self.width_latents = width_latents
        self.height_diffusion = height_diffusion_corrected
        self.width_diffusion = width_diffusion_corrected

    def _init_image2image(self):

        kwargs = {
            "pretrained_model_name_or_path": self.hf_model,
            "torch_dtype": torch.float16,
        }

        if self.use_lightning:
            unet = UNet2DConditionModel.from_config(self.hf_model, subfolder="unet").to(self.device, torch.float16)
            unet.load_state_dict(load_file(hf_hub_download(self.repo, self.ckpt), device=self.device))
            unet.to(self.device, torch.float16)

            kwargs = {
                "pretrained_model_name_or_path": self.hf_model,
                "unet": unet,
                "torch_dtype": torch.float16,
                "variant": "fp16",
            }

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(**kwargs)
        self._init_pipe(pipe)
        self.init_input_image_noise()

    def _init_text2image(self):

        kwargs = {
            "pretrained_model_name_or_path": self.hf_model,
            "torch_dtype": torch.float16,
        }

        if self.use_lightning:
            unet = UNet2DConditionModel.from_config(self.hf_model, subfolder="unet").to(self.device, torch.float16)
            unet.load_state_dict(load_file(hf_hub_download(self.repo, self.ckpt), device=self.device))
            unet.to(self.device, torch.float16)

            kwargs = {
                "pretrained_model_name_or_path": self.hf_model,
                "unet": unet,
                "torch_dtype": torch.float16,
                "variant": "fp16",
            }

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(**kwargs)
        self._init_pipe(pipe)

    def _init_pipe(self, pipe: StableDiffusionXLImg2ImgPipeline):
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)


        if "flash" in self.hf_model:
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing",
            )

        if self.use_lightning:
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing",
            )

        if self.use_tinyautoenc:
            pipe.vae = AutoencoderTiny.from_pretrained(
                'madebyollin/taesdxl',
                torch_device=self.device,
                torch_dtype=torch.float16,
            )
            pipe.vae = pipe.vae.to(self.device)

        # Load and fuse LoRAs if enabled
        if self.use_loras:
            self._load_and_fuse_loras(pipe)

        # Apply compilation if needed
        if self.do_compile:
            pipe.enable_xformers_memory_efficient_attention()
            config = CompilationConfig.Default()
            config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            config.enable_jit = True
            config.enable_jit_freeze = True
            config.enable_cnn_optimization = True
            config.preserve_parameters = True
            config.prefer_lowp_gemm = True
            pipe = compile(pipe, config)
        
        self.pipe = pipe
        self.set_latents()
        
    def _load_and_fuse_loras(self, pipe: StableDiffusionXLImg2ImgPipeline):
        """
        Helper method to load and fuse multiple LoRAs according to the configurations
        
        Args:
            pipe: The diffusion pipeline to which LoRAs will be applied
        """
        if not self.lora_configs or len(self.lora_configs) == 0:
            return
            
        adapter_names = []
        adapter_weights = []
        
        # Load all configured LoRAs
        for config in self.lora_configs:
            repo_id = config["repo_id"]
            weight_name = config.get("weight_name", None)  # Optional
            adapter_name = config["adapter_name"]
            weight = config.get("weight", 1.0)  # Default weight is 1.0
            
            # Load the LoRA weights
            if weight_name:
                pipe.load_lora_weights(repo_id, weight_name=weight_name, adapter_name=adapter_name)
            else:
                pipe.load_lora_weights(repo_id, adapter_name=adapter_name)
                
            adapter_names.append(adapter_name)
            adapter_weights.append(float(weight))
        
        # Set all adapters with their respective weights
        if adapter_names:
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            # Fuse all LoRAs
            pipe.fuse_lora(adapter_names=adapter_names, lora_scale=self.lora_scale)
            # Unload weights to free memory
            pipe.unload_lora_weights()
            
    def set_lora_config(self, lora_configs, enable=True):
        """
        Update LoRA configurations and optionally enable/disable LoRA usage
        
        Args:
            lora_configs: List of dictionaries with LoRA configurations
            enable: Whether to enable LoRA usage
        """
        self.lora_configs = lora_configs
        self.use_loras = enable
        
        # Reload the pipe if it already exists to apply new LoRA settings
        if hasattr(self, 'pipe'):
            if self.use_loras:
                self._load_and_fuse_loras(self.pipe)
            # Note: If disabling LoRAs, current LoRAs remain fused until pipe is reinitialized

    def set_latents(self, latents=None):
        if latents is None:
            latents = self.get_latents()
        self.latents = latents

    def get_latents(self, force_seed=None):
        if force_seed is None:
            torch.manual_seed(self.seed)
        else:
            torch.manual_seed(force_seed)
        
        latents = torch.randn((1, 4, self.height_latents, self.width_latents)).half().to(self.device)
        return latents

    def set_num_inference_steps(self, num_inference_steps, force_minimum_strength=False):
        self.num_inference_steps = int(num_inference_steps)
        
        if force_minimum_strength:
            if num_inference_steps > 1:
                strength = 1/num_inference_steps + 0.0001
            else:
                strength = 1
            self.strength = float(strength)

    def set_guidance_scale(self, guidance_scale):
        self.guidance_scale = float(guidance_scale)
        
    def set_strength(self, strength):
        self.strength = float(strength)

    def init_input_image_noise(self):
        noise_image = np.random.randn(self.height_diffusion, self.width_diffusion, 3)
        noise_image = ((noise_image - noise_image.min()) / (noise_image.max() - noise_image.min()) * 255).astype(np.uint8)
        self.set_input_image(noise_image)

    def set_input_image(self, image_init):
        if not isinstance(image_init, Image.Image):
            if image_init.dtype != np.uint8:
                image_init = np.round(image_init)
                image_init = np.clip(image_init, 0, 255)
                image_init = image_init.astype(np.uint8)
            image_init = Image.fromarray(image_init)
        
        width, height = image_init.size
        if height != self.height_diffusion or width != self.width_diffusion:
            image_init = image_init.resize((self.width_diffusion, self.height_diffusion))
        self.image_init = image_init

    def set_embeddings(self, *args):
        if len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 4:
            self.embeds = args[0]
        elif len(args) == 4:
            self.embeds = list(args)
        else:
            raise ValueError("Invalid input. Please provide either four separate embeddings or a list containing the four embeddings.")

    def build_kwargs(self, kwargs_override=None):
        kwargs = {}
        
        # Add prompt-related fields if embeddings are set
        if self.embeds is not None:
            kwargs.update({
                'prompt_embeds': self.embeds[0],
                'negative_prompt_embeds': self.embeds[1],
                'pooled_prompt_embeds': self.embeds[2],
                'negative_pooled_prompt_embeds': self.embeds[3]
            })
            
        kwargs.update({
            'num_inference_steps': self.num_inference_steps,
            'guidance_scale': self.guidance_scale,
        })
        
        if self.use_image2image and self.image_init is not None:
            kwargs.update({
                'image': self.image_init,
                'strength': self.strength
            })

        # Override with any provided kwargs
        if kwargs_override is not None:
            kwargs.update(kwargs_override)
            
        return kwargs
        
    def build_cross_attention_kwargs(self, kwargs, cross_attention_kwargs_override=None):
        # In the simplified version, we ignore modulations but support other cross attention kwargs
        if cross_attention_kwargs_override is not None and len(cross_attention_kwargs_override) > 0:
            kwargs['cross_attention_kwargs'] = cross_attention_kwargs_override
        return kwargs

    def generate(self, kwargs_override=None, cross_attention_kwargs_override=None):
        if not self.do_diffusion:
            return self.image_init

        # Build kwargs from attributes and overrides
        kwargs = self.build_kwargs(kwargs_override)
        kwargs = self.build_cross_attention_kwargs(kwargs, cross_attention_kwargs_override)
        
        # Generate image using the diffusers pipeline
        torch.manual_seed(self.seed)
        output = self.pipe(**kwargs)
        img_diffusion = output.images[0]
        
        return img_diffusion