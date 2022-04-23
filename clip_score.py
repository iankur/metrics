import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
import collections.abc as container_abcs
from tqdm import tqdm, trange
from glob import glob
import argparse

import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions_dir', type=str)
    parser.add_argument('--samples_dir', type=str)
    args = parser.parse_args()
    
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    all_logits = []
    
    ids = [d for d in os.listdir(args.samples_dir) if not d.startswith('.')]
    
    for id in tqdm(ids, leave=False):    
        prompt_path = os.path.join(args.captions_dir, f"{id}.txt")
        prompt = open(prompt_path).read().strip()
        
        image_paths = glob(os.path.join(args.samples_dir, id, '*.png'))
        images = [Image.open(image_path) for image_path in image_paths]
        
        with torch.no_grad():
            inputs = processor(text=[prompt], images=images, return_tensors="pt", 
                               padding='max_length', max_length=77, truncation=True)
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            logits_per_image = outputs.logits_per_image
            logits = logits_per_image.cpu().numpy().flatten()

        all_logits.append(logits)

    all_logits = np.stack(all_logits, axis=0)
    print(f"the average max score is: {all_logits.max(1).mean()}")
    print(f"the average mean score is: {all_logits.mean()}")
    
if __name__ == '__main__':
    main()
