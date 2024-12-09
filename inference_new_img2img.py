from io import BytesIO
from pathlib import Path
import os
import time
import requests
import torch
import flask
from diffusers import FluxControlImg2ImgPipeline
from huggingface_hub import snapshot_download, login
from PIL import Image
from PIL import ImageOps
from flask import Flask, request, send_file
from diffusers.utils import load_image
from controlnet_aux import *
import numpy as np
import cv2
import signal
import sys
# from skimage.morphology import skeletonize
import runpod
# Set up environment variables and paths
MODEL_DIR = "/workspace/models"  # Path to store model weights
pipe=None
# Login to Hugging Face Hub
# login(token="YOUR_HF_TOKEN")  # Replace with your actual token
torch.backends.cuda.matmul.allow_tf32 = True
# token = os.getenv("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") 
# if token:
#     login(token)
# else:
#     print("Warning: No Hugging Face token found. You may encounter rate limits.")
login('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')
# Download models and save them to the specified directory
def download_models():
    ignore = ["*.bin", "*.onnx_data"]
    snapshot_download("black-forest-labs/FLUX.1-Canny-dev", ignore_patterns=ignore, local_dir=MODEL_DIR)


# Load models from the specified directory
def load_models():
    from transformers.utils import move_cache
    move_cache()
    
    pipe = FluxControlImg2ImgPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16
    )
   
    pipe.to(device="cuda", dtype=torch.bfloat16)
    return pipe
def preprocess_image(init_image, method):
    # filename = image_url.split('/')[-1].split('.')[0]
    img = init_image
    
    # Initialize detectors
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
    lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    teed = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
    anyline = AnylineDetector.from_pretrained("TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline")
    canny = CannyDetector()
    # Process image based on the method key
    processed_image = None
    if method == 'hed':
        processed_image = hed(img)
    elif method == 'pidi':
        processed_image = pidi(img, safe=True)
    elif method == 'lineart':
        processed_image = lineart(img, coarse=True)
    elif method == 'lineart_anime':
        processed_image = lineart_anime(img)
    elif method == 'teed':
        processed_image = teed(img, detect_resolution=1024)
    elif method == 'anyline':
        processed_image = anyline(img, detect_resolution=1280)
    elif method=='canny':
        
        processed_image_image = canny(img, low_threshold=100, high_threshold=200,detect_resolution=1024, image_resolution=1024)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save processed image
    # processed_image.save('/workspace/canny_images/%s_%s.png' % (method, filename))
    
    return processed_image
def inference(pipe,prompt: str, init_image=None,method='canny'):
    import random
    seed = random.randint(0, 2**16 - 1)
    start_time = time.time()

    if init_image:
        # Read the uploaded file's content
        image_bytes = init_image.read()

        # Convert bytes data to a PIL Image
        try:
            init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            
            # Save the image to the local disk
            # init_image.save("/workspace/image.png")  # Specify the desired path and filename
                
        except Exception as e:
            print(f"Error opening image: {e}")
            return "Invalid image file", 400
    else:
        init_image = None

   # Step 1: Apply Canny edge detection to the initial image
    control_image = preprocess_image(init_image,method)
    # control_image.save('/workspace/canny.png')
    
    # Step 3: Clean up edges with a threshold
    edges_array = np.array(control_image.convert("L"))
    threshold = 128
    edges_array = (edges_array > threshold).astype(np.uint8) * 255  # Binary threshold for cleaner edges
    
    # Step 4: Convert the cleaned-up edges back to a PIL image
    clean_edges = Image.fromarray(edges_array)
    
    # Step 5: Invert the edges and convert to RGB
    inverted_image = ImageOps.invert(clean_edges).convert("RGB")
    
    # Step 6: Save the inverted image
    # inverted_image.save('/workspace/inverted_canny.png') 
    # Inference step
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        image = pipe(
            prompt=prompt,
            image=inverted_image,
            control_image=control_image,
            generator=torch.Generator().manual_seed(int(seed)),
            strength=0.8,
            num_inference_steps=50,
            guidance_scale=float(30.0),
            height=int(1024),
            width=int(1024),
            max_sequence_length=256
        ).images[0]

    

    end_time = time.time()
    print(f"Inference took {end_time - start_time} seconds")
    
    # Instead of returning the original generated image, return the inverted Canny image of the output
    byte_stream = BytesIO()
    image.save(byte_stream, format="PNG")
    byte_stream.seek(0)
    return send_file(byte_stream, mimetype="image/png")

download_models()  # Uncomment to download models
pipe=load_models()

# Set up Flask app
def handler(event):
    prompt = request.form.get("prompt")
    image_init = request.files.get('image_file')
    key = request.form.get('api_key')
    method = request.form.get('method')
    if key != "#$$$jjkj56gw397987":  # Replace with a more secure API key management
        return "Unauthorized", 401
    
    if not prompt:
        return "Missing prompt", 400 

    output_image = inference(pipe,prompt, image_init,method)
    return output_image



runpod.serverless.start({"handler": handler})