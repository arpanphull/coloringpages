from io import BytesIO
from pathlib import Path
import os
import time
import requests
import torch
from diffusers import FluxPipeline
from huggingface_hub import snapshot_download
from PIL import Image
import gc
import runpod

# Set up environment variables and paths
MODEL_DIR = "models"  # Path to store model weights

# Configure GPU settings
torch.backends.cuda.matmul.allow_tf32 = True


# Load models from the specified directory
def load_models():
    from transformers.utils import move_cache
    move_cache()
    pipe = FluxPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16
    )
    pipe.load_lora_weights(MODEL_DIR, weight_name="Hyper-FLUX.1-dev-8steps-lora.safetensors", adapter_name="bd")
    pipe.load_lora_weights(MODEL_DIR, weight_name='2eee4f582d2844b19c6c3a9eb454213a_pytorch_lora_weights.safetensors', adapter_name='custom')
    pipe.set_adapters(["bd", "custom"], adapter_weights=[0.125, 0.8])
    pipe.fuse_lora()
    pipe.to(device="cuda", dtype=torch.bfloat16)
    return pipe

# Perform inference
def inference(pipe, prompt: str):
    import random
    seed = random.randint(0, 2**16 - 1)
    start_time = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        image = pipe(
            prompt=[prompt],
            generator=torch.Generator().manual_seed(int(seed)),
            num_inference_steps=8,
            guidance_scale=float(3.5),
            height=int(1024),
            width=int(1024),
            max_sequence_length=256
        ).images[0]
    end_time = time.time()
    print(f"Inference took {end_time - start_time} seconds")
    byte_stream = BytesIO()
    image.save(byte_stream, format="PNG")
    image_bytes = byte_stream.getvalue()
    return image_bytes

# Initialize models and clear GPU cache
gc.collect()
torch.cuda.empty_cache()
pipe = load_models()

# Define RunPod handler
def handler(event):
    prompt = event.get("input", {}).get("prompt")
    api_key = event.get("input", {}).get("api_key")

    # Check for API key authorization
    if api_key != '#$$$jjkj56gw397987':
        return {"error": "Unauthorized", "status": 401}

    if not prompt:
        return {"error": "Prompt is required", "status": 400}

    # Generate the image
    output_image_bytes = inference(pipe, prompt)
    return {"status": 200, "image_bytes": output_image_bytes}

# Start RunPod
runpod.serverless.start({"handler": handler})
