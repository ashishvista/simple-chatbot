import requests
import json
import os
from pdf2image import convert_from_path
import base64
import io

OLLAMA_HOST = "http://44.204.80.225:11434"
MODEL_NAME = "qwen2.5vl:32b"

def pdf_to_base64_images(pdf_path):
    """
    Converts each page of the PDF to a base64-encoded PNG image.
    Also saves each PNG to disk in the same directory as the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        raise RuntimeError(
            "pdf2image requires Poppler to be installed and in your PATH. "
            "Install Poppler:\n"
            "  - On macOS: brew install poppler\n"
            "  - On Ubuntu: sudo apt-get install poppler-utils\n"
            "  - On Windows: Download from http://blog.alivate.com.au/poppler-windows/ and add to PATH\n"
            f"Original error: {e}"
        )
    base64_images = []
    pdf_dir = os.path.dirname(pdf_path)
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    for idx, page in enumerate(pages, 1):
        png_filename = os.path.join(pdf_dir, f"{pdf_base}_page{idx}.png")
        page.save(png_filename, format="PNG")
        buffered = io.BytesIO()
        page.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        base64_images.append(img_str)
    return base64_images

def extract_data_with_qwen2(images_base64, prompt_instructions=None):
    """
    Calls Ollama's /api/chat endpoint with Minicpm-v:8b to extract data as markdown.
    """
    prompt = (
        "Perform OCR on the attached document images and extract all relevant data as accurately as possible. "
        "Return the extracted data in a clear and structured markdown format, using headings and tables where appropriate."
    )
    if prompt_instructions:
        prompt += f" {prompt_instructions}"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_base64  # Attach images directly to the user message
            }
        ],
        "stream": False
    }
    print("Request payload:", json.dumps(payload)[:1000], "..." if len(json.dumps(payload)) > 1000 else "")  # Print first 1000 chars for brevity
    url = f"{OLLAMA_HOST}/api/chat"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if not response.ok:
        raise RuntimeError(f"Ollama chat failed: {response.text}")
    try:
        resp_json = response.json()
    except Exception as e:
        print("Raw chat response:", response.text)
        raise RuntimeError(f"Failed to parse JSON from chat response: {e}")
    return resp_json.get("message") or resp_json.get("response") or resp_json

def extract_pdf_to_markdown(pdf_path, prompt_instructions=None):
    """
    Complete pipeline: convert PDF to images, extract data using Minicpm-v:8b, return markdown.
    """
    images_base64 = pdf_to_base64_images(pdf_path)
    result = extract_data_with_qwen2(images_base64, prompt_instructions)
    return result

# Example usage:
if __name__ == "__main__":
    # Replace with your PDF path
    pdf_path = "examples/example-mri.pdf"
    prompt_instructions = "Extract all fields as accurately as possible."
    extracted_markdown = extract_pdf_to_markdown(pdf_path, prompt_instructions)
    print(extracted_markdown)
