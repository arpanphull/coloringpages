from io import BytesIO
from pathlib import Path
import os
import time
import base64
import requests
import torch
from diffusers import FluxControlImg2ImgPipeline
from huggingface_hub import snapshot_download, login
from PIL import Image, ImageOps
import numpy as np
import runpod
# Import detectors from controlnet_aux
from controlnet_aux import (
    HEDdetector,
    PidiNetDetector,
    LineartDetector,
    LineartAnimeDetector,
    TEEDdetector,
    AnylineDetector,
    CannyDetector
)

# Set up environment variables and paths
MODEL_DIR = "/workspace/models"  # Path to store model weights
pipe = None

# Login to Hugging Face Hub
# Replace with your actual Hugging Face token
HF_TOKEN = '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'
login(HF_TOKEN)

# Download models and save them to the specified directory
def download_models():
    ignore = ["*.bin", "*.onnx_data"]
    snapshot_download(
        "black-forest-labs/FLUX.1-Canny-dev",
        ignore_patterns=ignore,
        local_dir=MODEL_DIR
    )

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

# Preprocess the input image based on the specified method
def preprocess_image(init_image, method):
    img = init_image

    # Initialize detectors
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
    lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    teed = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
    anyline = AnylineDetector.from_pretrained(
        "TheMistoAI/MistoLine",
        filename="MTEED.pth",
        subfolder="Anyline"
    )
    canny = CannyDetector()

    # Process image based on the method key
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
    elif method == 'canny':
        processed_image = canny(img, low_threshold=100, high_threshold=200, detect_resolution=1024, image_resolution=1024)
    else:
        raise ValueError(f"Unknown method: {method}")

    return processed_image

# Inference function
def inference(pipe, prompt: str, init_image: Image.Image = None, method: str = 'canny'):
    import random
    seed = random.randint(0, 2**16 - 1)
    start_time = time.time()

    if init_image:
        # Convert the image to RGB if it's not
        init_image = init_image.convert("RGB")
    else:
        init_image = None

    # Apply the specified preprocessing method
    control_image = preprocess_image(init_image, method)

    # Clean up edges with a threshold
    edges_array = np.array(control_image.convert("L"))
    threshold = 128
    edges_array = (edges_array > threshold).astype(np.uint8) * 255  # Binary threshold for cleaner edges

    # Convert the cleaned-up edges back to a PIL image
    clean_edges = Image.fromarray(edges_array)

    # Invert the edges and convert to RGB
    inverted_image = ImageOps.invert(clean_edges).convert("RGB")

    # Inference step
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated_image = pipe(
            prompt=prompt,
            image=inverted_image,
            control_image=control_image,
            generator=torch.Generator().manual_seed(int(seed)),
            strength=0.8,
            num_inference_steps=50,
            guidance_scale=30.0,
            height=1024,
            width=1024,
            max_sequence_length=256
        ).images[0]

    end_time = time.time()
    print(f"Inference took {end_time - start_time} seconds")

    # Convert the generated image to base64
    byte_stream = BytesIO()
    generated_image.save(byte_stream, format="PNG")
    byte_stream.seek(0)
    base64_image = base64.b64encode(byte_stream.read()).decode('utf-8')

    return base64_image

# Initialize models at startup
download_models()  # Ensure models are downloaded
pipe = load_models()

# Handler function for RunPod serverless
def handler(event):
    #try:
    input_data = event.get("input", {})
    prompt = input_data.get("prompt")
    api_key = input_data.get("api_key")
    method = input_data.get("method", "canny")
    image_base64 = input_data.get("image_file")  # Expecting base64-encoded image string

    # Validate API key
    if api_key != "#$$$jjkj56gw397987":  # Replace with a more secure solution
        return {"error": "Unauthorized"}, 401

    # Validate prompt
    if not prompt:
        return {"error": "Missing prompt"}, 400

    # Decode the base64 image if provided
    init_image = None
    if image_base64:
        try:
            image_bytes = base64.b64decode(image_base64)
            init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"error": f"Invalid image data: {str(e)}"}, 400

    # Run inference
    output_image_base64 = inference(pipe, prompt, init_image, method)

    # Return the base64-encoded image
    return {"image_data": output_image_base64}, 200

    # except Exception as e:
    #     # Catch any unexpected errors
    #     return {"error": str(e)}, 500

# Start the RunPod serverless function
runpod.serverless.start({"handler": handler})
