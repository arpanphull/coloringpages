import base64
from io import BytesIO
from PIL import Image, ImageOps
import runpod
from diffusers import FluxControlImg2ImgPipeline
# Assume `inference` function and `pipe` are defined and loaded elsewhere
# from your existing code

# Handler function for RunPod serverless
def handler(event):
    try:
        # Extract inputs from the event dictionary
        input_data = event.get("input", {})
        prompt = input_data.get("prompt")
        api_key = input_data.get("api_key")
        method = input_data.get("method", "canny")  # Optional, default to 'canny'
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
        # output_image = inference(pipe, prompt, init_image, method)
        output_image=init_image
        # Convert the output image to base64
        byte_stream = BytesIO()
        output_image.save(byte_stream, format="PNG")
        byte_stream.seek(0)
        output_base64 = base64.b64encode(byte_stream.read()).decode('utf-8')

        # Return the base64-encoded image
        return {"image_data": output_base64}, 200

    except Exception as e:
        # Catch any unexpected errors
        return {"error": str(e)}, 500

# Start the RunPod serverless function
runpod.serverless.start({"handler": handler})
