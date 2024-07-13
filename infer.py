"""
Modified from:
    Break-A-Scene: https://github.com/google/break-a-scene
"""

import argparse

from diffusers import DiffusionPipeline, DDIMScheduler
import torch

import random
import numpy as np
from PIL import Image
from ptp_utils import load_learned_embed
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

@torch.no_grad()
def infer_with_embed(embed_path, pretrained, prompt, num_samples=2, num_rows=1, return_all_images=False, disable_progress_bar=False):
    text_encoder, tokenizer = load_learned_embed(embed_path, pretrained)

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    ).to("cuda")

    if disable_progress_bar:
        pipe.set_progress_bar_config(leave=False)
        pipe.set_progress_bar_config(disable=True)
        
    all_images = []
    for _ in range(num_rows):
        images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5).images
        all_images.extend(images)

    grid = image_grid(all_images, num_rows, num_samples)
    
    if return_all_images:
        return grid, all_images
    else:   
        return grid

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=2,
        help="Number of rows to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Number of rows to generate.",
    )
    parser.add_argument(
        "--save_full_model",
        action='store_true',
        help="Whether to store the entire model (or only save the embeddings).",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        default=None,
        help="Path to the learned embedding.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to the full model checkpoint.",
    ) 
    parser.add_argument(
        "--pretrained",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="[stabilityai/stable-diffusion-2, stabilityai/stable-diffusion-2-base, CompVis/stable-diffusion-v1-4, runwayml/stable-diffusion-v1-5]",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    ) 
    args = parser.parse_args()

    return args
 
if __name__ == "__main__":
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    grid, all_images = infer_with_embed(args.embed_path, args.pretrained, args.prompt, num_samples=args.num_samples, num_rows=args.num_rows, return_all_images=True)
    
    image_path = os.path.join(args.save_path, args.prompt.replace(' ', '-'))
    
    if not os.path.exists(image_path):
        from pathlib import Path
        Path(image_path).mkdir(parents=True, exist_ok=True)
        
    for idx, img in enumerate(all_images):
        img.save(os.path.join(image_path, '{}.png'.format(idx)))
        
    grid.save(os.path.join(args.save_path, args.prompt.replace(' ', '-') + '.png'))