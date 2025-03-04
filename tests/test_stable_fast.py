import time
import torch
import os
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForImage2Image,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import load_image, make_image_grid
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)

def save_image(image, output_dir="/media/monsterdrive/g_test/rtd/tests/output"):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to: {output_path}")

def load_sdxl_lightning_img2img(steps=4):
    """Load SDXL Lightning model for img2img"""
    # Base model to load configuration from
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    # SDXL Lightning repo
    repo = "ByteDance/SDXL-Lightning"
    # Select checkpoint based on number of steps
    ckpt_map = {
        1: "sdxl_lightning_1step_unet.safetensors",
        2: "sdxl_lightning_2step_unet.safetensors",
        4: "sdxl_lightning_4step_unet.safetensors",
        8: "sdxl_lightning_8step_unet.safetensors"
    }
    ckpt = ckpt_map.get(steps, "sdxl_lightning_4step_unet.safetensors")
    
    print(f"Loading SDXL Lightning with {steps} steps checkpoint...")
    
    # Load and replace the UNet
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    
    # Create img2img pipeline with the custom UNet
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        base, 
        unet=unet,
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to("cuda")
    
    # Lightning models use Euler discrete scheduler with trailing timesteps
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="trailing"
    )
    
    pipe.safety_checker = None
    return pipe

# Choose which implementation to use
use_lightning = False
steps = 4  # Choose from 1, 2, 4, or 8 steps

if use_lightning:
    pipe = load_sdxl_lightning_img2img(steps)
else:
    # Original SDXL-Flash implementation
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "sd-community/sdxl-flash",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    pipe.safety_checker = None

# Apply stable-fast optimizations
pipe.enable_xformers_memory_efficient_attention()
config = CompilationConfig.Default()
config.enable_xformers = True
config.enable_triton = True
config.enable_cuda_graph = True
config.enable_jit = True
config.enable_jit_freeze = True
# config.trace_scheduler = True # Makes this poor quality
config.enable_cnn_optimization = True
config.preserve_parameters = True
config.prefer_lowp_gemm = True

model = compile(pipe, config)

kwarg_inputs = dict(
    image=load_image("/media/monsterdrive/g_test/rtd/tests/output/generated_image_20250302_122818.png"),
    prompt='(masterpiece:1,2), best quality, masterpiece, best detailed face, a beautiful girl',
    strength=0.6,
    # For Lightning models, guidance_scale=0 is recommended
    guidance_scale=0.0 if use_lightning else 1.2,
    height=512,
    width=512,
    # Use the correct number of steps for the model
    num_inference_steps=steps if use_lightning else 10,
    num_images_per_prompt=1,
)

# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
print("Warming up model with initial runs...")
for i in range(3):
    output_image = model(**kwarg_inputs).images[0]
    print(f"Warmup run {i+1} complete")

# Let's see it!
# Note: Progress bar might work incorrectly due to the async nature of CUDA.
print("\nRunning inference...")
begin = time.time()
output_image = model(**kwarg_inputs).images[0]
inference_time = time.time() - begin
print(f'Inference time: {inference_time:.3f}s')

# Save the generated image
save_image(output_image)

# Let's view it in terminal!
from sfast.utils.term_image import print_image
print_image(output_image, max_width=80)