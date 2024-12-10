import base64
import json
import argparse
import os
import sys
import textwrap
from PIL import Image

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

def generate_json(prompt, api_key, method, image_base64):
    """
    Generates the JSON structure with the provided parameters.

    Args:
        prompt (str): The prompt for image generation.
        api_key (str): API key for authorization.
        method (str): Preprocessing method.
        image_base64 (str): Base64-encoded image string.

    Returns:
        dict: JSON object as a Python dictionary.
    """
    data = {
        "input": {
            "prompt": prompt,
            "api_key": api_key,
            "method": method,
            "image_file": image_base64
        }
    }
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate test_input.json with base64-encoded image.")
    parser.add_argument('--image', '-i', required=True, help="Path to the image file to encode.")
    parser.add_argument('--output', '-o', help="Path to save the generated JSON. If not provided, prints to console.")
    
    args = parser.parse_args()
    
    image_path = args.image
    output_path = args.output

    # Customize your prompt, API key, and method here
    prompt = "A serene landscape with mountains and a river at sunset"
    api_key = "#$$$jjkj56gw397987"  # Replace with your actual API key
    method = "canny"

    # Encode the image to base64
    image_base64 = encode_image_to_base64(image_path)

    # Generate the JSON structure
    json_data = generate_json(prompt, api_key, method, image_base64)

    # Convert the dictionary to a JSON string with indentation for readability
    json_str = json.dumps(json_data, indent=2)

    # Optionally, wrap the base64 string with line breaks for readability (optional)
    # Uncomment the following lines if you want the base64 string to have line breaks every 76 characters
    # base64_wrapped = '\n'.join(textwrap.wrap(image_base64, 76))
    # json_data["input"]["image_file"] = base64_wrapped
    # json_str = json.dumps(json_data, indent=2)

    if output_path:
        try:
            with open(output_path, "w") as json_file:
                json_file.write(json_str)
            print(f"JSON data successfully written to '{output_path}'.")
        except Exception as e:
            print(f"Error writing JSON to file: {e}")
            sys.exit(1)
    else:
        print(json_str)

if __name__ == "__main__":
    main()
