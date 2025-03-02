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
)

from diffusers.utils import load_image, make_image_grid

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

def load_model(use_img2img=True):
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     'runwayml/stable-diffusion-v1-5',
    #     torch_dtype=torch.float16)

    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if use_img2img:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "sd-community/sdxl-flash",
            torch_dtype=torch.float16,
        ).to("cuda")
    else:
        # Load model.
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "sd-community/sdxl-flash",
            torch_dtype=torch.float16,
        ).to("cuda")

    # Ensure sampler uses "trailing" timesteps.
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    pipe.safety_checker = None
    pipe.to(torch.device('cuda'))
    return pipe

pipe = load_model()

config = CompilationConfig.Default()

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
    guidance_scale=1.2,
    height=512,
    width=512,
    num_inference_steps=10,
    num_images_per_prompt=1,
)

# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
for _ in range(3):
    output_image = model(**kwarg_inputs).images[0]

# Let's see it!
# Note: Progress bar might work incorrectly due to the async nature of CUDA.
begin = time.time()
output_image = model(**kwarg_inputs).images[0]
print(f'Inference time: {time.time() - begin:.3f}s')

# Save the generated image
save_image(output_image)

# Let's view it in terminal!
from sfast.utils.term_image import print_image

print(type(output_image))
print_image(output_image, max_width=80)

# if "flash" in self.hf_model:
#     pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
#         pipe.scheduler.config,
#         timestep_spacing="trailing",
#     )