import argparse
import time
import numpy as np
import torch
import lunar_tools as lt
from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
from rtd.sdxl_turbo.simple_diffusion_engine import SimpleDiffusionEngine
from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
from diffusers.utils import load_image, make_image_grid
import os
from pathlib import Path

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

def test_threaded():
    pass
#     height_diffusion = 512
#     width_diffusion = 512
#     de = DiffusionEngine(use_image2image=True, height_diffusion_desired=height_diffusion, width_diffusion_desired=width_diffusion)
#     em = EmbeddingsMixer(de.pipe)
#     embeds = em.encode_prompt("photo of a man")
#     de_threaded = DiffusionEngineThreaded(de)
#     print("Threaded test completed successfully")

def test_t2i():

    height_diffusion = 768
    width_diffusion = 1024

    hf_model = "stabilityai/sdxl-turbo"
    hf_model = "sd-community/sdxl-flash"
    hf_model = "stabilityai/stable-diffusion-xl-base-1.0"

    de = DiffusionEngine(
        hf_model=hf_model,
        use_image2image=False,
        height_diffusion_desired=height_diffusion,
        width_diffusion_desired=width_diffusion,
        do_compile=True,
    )

    em = EmbeddingsMixer(de.pipe)
    # de_txt = DiffusionEngineThreaded(de_txt)
    embeds = em.encode_prompt("(masterpiece:1,2), best quality, masterpiece, best detailed face, a beautiful girl")
    de.set_embeddings(embeds)
    
    renderer = lt.Renderer(
        width=width_diffusion,
        height=height_diffusion,
        backend='opencv',
        do_fullscreen=False,
    )
    # midi_input = lt.MidiInput(device_name="akai_midimix")
    
    kwargs_override = {
        "num_inference_steps": 4,
    }

    print("Press Ctrl+C to exit")
    while True:
        # num_inference_steps = int(midi_input.get("A0", val_min=1, val_max=5))
        img = de.generate(kwargs_override=kwargs_override)
        renderer.render(img)

def test_i2i():

    height_diffusion = 512
    width_diffusion = 512

    hf_model = "sd-community/sdxl-flash"
    # hf_model = "stabilityai/stable-diffusion-xl-base-1.0"
    # hf_model = "stabilityai/sdxl-turbo"

    de = SimpleDiffusionEngine(
        # hf_model=hf_model,
        use_image2image=True,
        height_diffusion_desired=height_diffusion,
        width_diffusion_desired=width_diffusion,
        do_compile=True,
        use_lightning=True,
    )

    em = EmbeddingsMixer(de.pipe)
    # de_txt = DiffusionEngineThreaded(de_txt)
    embeds = em.encode_prompt("(masterpiece:1,2), best quality, masterpiece, best detailed face, a beautiful girl")
    img_init = load_image("/media/monsterdrive/g_test/rtd/tests/output/generated_image_20250302_122818.png")
    de.set_input_image(img_init)
    de.set_embeddings(embeds)

    de.set_guidance_scale(0.0) #0.5
    de.set_strength(0.6) #1 / self.de_img.num_inference_steps + 0.00001)
    de.set_num_inference_steps(2)
    
    renderer = lt.Renderer(
        width=width_diffusion,
        height=height_diffusion,
        backend='opencv',
        do_fullscreen=False,
    )
    # midi_input = lt.MidiInput(device_name="akai_midimix")
    
    # kwargs_override = {
    #     # "num_inference_steps": 4,
    #     # "guidance_scale": 0,
    #     # "strength": 0.6,
    # }

    img = de.generate(kwargs_override=None)
    save_image(image=img)

    # print("Press Ctrl+C to exit")
    # while True:
    #     # num_inference_steps = int(midi_input.get("A0", val_min=1, val_max=5))
    #     img = de.generate(kwargs_override=kwargs_override)
    #     renderer.render(img)

def test_prompt_sequence():
    height_diffusion = 704
    width_diffusion = 1024
    de_txt = DiffusionEngine(use_image2image=False, height_diffusion_desired=height_diffusion, width_diffusion_desired=width_diffusion)
    em = EmbeddingsMixer(de_txt.pipe)
    
    renderer = lt.Renderer(width=width_diffusion, height=height_diffusion, backend='opencv', do_fullscreen=False)
    midi_input = lt.MidiInput(device_name="akai_midimix")

    prompts = [
        "a blue dog and a red monkey",
        "a yellow cat and a green bird",
        "a purple elephant and a pink rabbit",
    ]
    
    for prompt in prompts:
        time.sleep(1)
        num_inference_steps = int(midi_input.get("A0", val_min=1, val_max=5))
        kwargs_override = {"num_inference_steps": num_inference_steps}

        embeds = em.encode_prompt(prompt=prompt)
        de_txt.set_embeddings(embeds)

        img = de_txt.generate(kwargs_override=kwargs_override)
        renderer.render(img)

def test_decoder_embeddings():
    height_diffusion = 512
    width_diffusion = 512
    de_txt = DiffusionEngine(use_image2image=False, height_diffusion_desired=height_diffusion, width_diffusion_desired=width_diffusion)
    em = EmbeddingsMixer(de_txt.pipe)
    
    de_txt.set_num_inference_steps = 2
    
    midi_input = lt.MidiInput(device_name="akai_lpd8")
    
    main_prompt = "photo of a landscape"
    list_decoder_embeds = [main_prompt, "autumn", "volcanic"]
    
    embeds = em.encode_prompt(main_prompt)
    de_txt.set_embeddings(embeds)
    
    em.encode_and_store_prompts(list_decoder_embeds)
    renderer = lt.Renderer(width=width_diffusion, height=height_diffusion, backend='opencv', do_fullscreen=False)
    
    ms = lt.MovieSaver("decoder.mp4", fps=16, crf=28)
    
    print("Press Ctrl+C to exit and save video")
    try:
        while True:
            list_weights = []
            list_weights.append(midi_input.get("E0", val_min=0, val_max=1, val_default=0))
            list_weights.append(midi_input.get("F0", val_min=0, val_max=1, val_default=0))
            main_weights = 1 - np.sum(np.asarray(list_weights))
            list_weights.insert(0, main_weights)
            embeds_decoder_scaled = em.blend_stored_embeddings(list_weights)
            de_txt.set_decoder_embeddings(embeds_decoder_scaled)
            img = de_txt.generate()
            renderer.render(img)
            ms.write_frame(img)
    except KeyboardInterrupt:
        ms.finalize()
        print("Video saved as decoder.mp4")

def main():
    parser = argparse.ArgumentParser(description='Test different DiffusionEngine functionalities')
    parser.add_argument('test_type', choices=['threaded', 'i2i', 't2i', 'sequence', 'decoder'],
                      help='Which test to run: threaded, override (kwargs), sequence (prompts), or decoder (embeddings)')
    
    args = parser.parse_args()
    
    test_functions = {
        'threaded': test_threaded,
        'i2i': test_i2i,
        't2i': test_t2i,
        'sequence': test_prompt_sequence,
        'decoder': test_decoder_embeddings
    }
    
    try:
        test_functions[args.test_type]()
    except KeyboardInterrupt:
        print("\nTest terminated by user")

if __name__ == '__main__':
    main()
