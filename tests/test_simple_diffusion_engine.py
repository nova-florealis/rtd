import time
import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path
from rtd.sdxl_turbo.simple_diffusion_engine import SimpleDiffusionEngine

def save_image(image, output_dir="/media/monsterdrive/g_test/rtd/tests/output"):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"simple_diffusion_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to: {output_path}")

def test_simple_diffusion_engine():
    print("Initializing SimpleDiffusionEngine...")
    
    # Initialize the SimpleDiffusionEngine
    engine = SimpleDiffusionEngine(
        height_diffusion_desired=512, 
        width_diffusion_desired=512,
        use_image2image=True,
        hf_model="sd-community/sdxl-flash",
        do_compile=False  # Set to True to enable optimizations
    )
    
    # Create random noise as initial image
    height, width = engine.height_diffusion, engine.width_diffusion
    noise_image = np.random.randn(height, width, 3)
    noise_image = ((noise_image - noise_image.min()) / (noise_image.max() - noise_image.min()) * 255).astype(np.uint8)
    noise_image = Image.fromarray(noise_image)
    
    # Set the input image
    engine.set_input_image(noise_image)
    
    # Create dummy embeddings
    # In a real scenario, these would come from a text encoder
    batch_size = 1
    seq_len = 77  # Standard CLIP sequence length
    hidden_dim = 1280  # SDXL dimension
    
    # Create random embeddings for testing
    torch.manual_seed(42)
    prompt_embeds = torch.randn(batch_size, seq_len, hidden_dim, device=engine.device).half()
    negative_prompt_embeds = torch.randn(batch_size, seq_len, hidden_dim, device=engine.device).half()
    pooled_prompt_embeds = torch.randn(batch_size, hidden_dim, device=engine.device).half()
    negative_pooled_prompt_embeds = torch.randn(batch_size, hidden_dim, device=engine.device).half()
    
    # Set the embeddings
    engine.set_embeddings(prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
    
    # Set generation parameters
    engine.set_num_inference_steps(4)
    engine.set_guidance_scale(0.0)
    engine.set_strength(0.75)
    engine.seed = 42
    
    # Warm up the model
    print("Warming up the model...")
    for i in range(2):
        _ = engine.generate()
        print(f"Warmup run {i+1} completed")
    
    # Generate the final image
    print("\nGenerating image...")
    start_time = time.time()
    output_image = engine.generate()
    end_time = time.time()
    
    print(f"Image generation completed in {end_time - start_time:.2f} seconds")
    
    # Save the output image
    save_image(output_image)
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_simple_diffusion_engine()