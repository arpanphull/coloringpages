import base64
import json
import argparse
import os
import sys
import requests
from PIL import Image
from io import BytesIO

def encode_image_to_base64(image_path):
    """
    Encodes an image to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        sys.exit(1)
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_str = encoded_bytes.decode('utf-8')
            return encoded_str
    except Exception as e:
        print(f"Error encoding image: {e}")
        sys.exit(1)

def decode_base64_to_image(base64_str, output_path):
    """
    Decodes a base64 string to an image and saves it.

    Args:
        base64_str (str): Base64-encoded image string.
        output_path (str): Path to save the decoded image.

    Returns:
        None
    """
    try:
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        image.save(output_path)
        print(f"Output image saved to '{output_path}'.")
    except Exception as e:
        print(f"Error decoding image: {e}")
        sys.exit(1)

def send_post_request(url, payload):
    """
    Sends a POST request to the specified URL with the given JSON payload.

    Args:
        url (str): The endpoint URL.
        payload (dict): The JSON payload to send.

    Returns:
        dict: The JSON response from the server.

    Raises:
        Exception: If the request fails or the server returns an error.
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(url)
        print(json.dumps(payload))
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_response = response.json()
                error_message = error_response.get("error", response.text)
            except json.JSONDecodeError:
                error_message = response.text
            raise Exception(f"Request failed with status {response.status_code}: {error_message}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Send image to RunPod endpoint.")
    parser.add_argument('--image', '-i', required=True, help="Path to the input image file.")
    parser.add_argument('--prompt', '-p', required=True, help="Prompt for image generation.")
    parser.add_argument('--method', '-m', default="canny", help="Preprocessing method (default: canny).")
    parser.add_argument('--api_key', '-k', default="#$$$jjkj56gw397987", help="API key for authorization.")
    parser.add_argument('--output', '-o', default="output_image.png", help="Path to save the output image.")
    parser.add_argument('--url', '-u', default="https://api.runpod.ai/v2/q0sqqutqjs76li/run", help="RunPod endpoint URL.")
    
    args = parser.parse_args()
    
    image_path = args.image
    prompt = args.prompt
    method = args.method
    api_key = args.api_key
    output_path = args.output
    url = args.url
    
    # Encode the image to base64
    print(f"Encoding image '{image_path}' to base64...")
    image_base64 = encode_image_to_base64(image_path)
    
    # Construct the JSON payload
    payload = {
        "input": {
            "prompt": prompt,
            "api_key": api_key,
            "method": method,
            "image_file": image_base64
        }
    }
    
    print(f"Sending POST request to '{url}'...")
    
    try:
        response_json = send_post_request(url, payload)
        print("POST request successful.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Handle the response
    if "image_data" in response_json:
        print("Received 'image_data' in response.")
        decode_base64_to_image(response_json["image_data"], output_path)
    elif "error" in response_json:
        print(f"Error from server: {response_json['error']}")
    else:
        print("Unexpected response format.")
        print(json.dumps(response_json, indent=2))

if __name__ == "__main__":
    main()
