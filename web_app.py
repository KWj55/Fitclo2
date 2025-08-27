import os
from flask import Flask, send_from_directory, jsonify, request
import torch
from PIL import Image
import numpy as np
from diffusers.image_processor import VaeImageProcessor
import time
from huggingface_hub import snapshot_download

from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker

# --- Configuration ---
# Get the absolute path of the directory where the script is located
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Define paths to the HTML and image directories
FFITING_HTML_DIR = os.path.join(PROJECT_ROOT, 'ffiting_html')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
RESULT_DIR = os.path.join(IMAGES_DIR, 'result')

# --- Model Loading (do this once) ---
print("Loading models, this may take a while...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download and cache model files from Hugging Face Hub
print("Downloading models from Hugging Face Hub... (This may take a while on first run)")
repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
print("Model files are located at:", repo_path)

# Load the main pipeline
pipeline = CatVTONPipeline(
    attn_ckpt_version="dresscode", # or "vitonhd"
    attn_ckpt=repo_path, # Use the downloaded repo path
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    weight_dtype=torch.float16,
    device=device,
    skip_safety_check=True
)

# Load the masker
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device,
)

# Create processors
vae_processor = VaeImageProcessor(vae_scale_factor=8)
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
print("Models loaded successfully.")

# --- Flask App Setup ---
app = Flask(__name__)

# --- Helper Functions ---
def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# --- Routes ---
@app.route('/')
def index():
    """Serve the main.html file from the ffiting_html directory."""
    return send_from_directory(FFITING_HTML_DIR, 'main.html')

@app.route('/images/<path:filename>')
def serve_images(filename):
    """
    Serve images from the images directory.
    This is matched before the general asset route.
    """
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/<path:filename>')
def serve_ffiting_assets(filename):
    """
    Serve other assets (like CSS, JS, other HTML files) from the ffiting_html directory.
    This acts as a catch-all for files not matched by other routes.
    """
    return send_from_directory(FFITING_HTML_DIR, filename)

# --- API Endpoints ---
@app.route('/api/get_image_list', methods=['GET'])
def get_image_list():
    """
    Dynamically scans for images and returns a list of paths.
    """
    image_type = request.args.get('type', 'person')
    
    image_paths = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    scan_dirs = []
    if image_type == 'person':
        scan_dirs = ["human/men", "human/women"]
    elif image_type == 'clothing':
        scan_dirs = ["cloth/overall", "cloth/upper"]
    
    for rel_dir in scan_dirs:
        full_dir_path = os.path.join(IMAGES_DIR, rel_dir)
        if not os.path.isdir(full_dir_path):
            print(f"Warning: Directory not found, skipping: {full_dir_path}")
            continue
        
        for filename in sorted(os.listdir(full_dir_path)):
            if os.path.splitext(filename)[1].lower() in supported_extensions:
                web_path = os.path.join('images', rel_dir, filename).replace("\\", "/")
                image_paths.append(f'/{web_path}')
                
    return jsonify(image_paths)

@app.route('/api/fitting', methods=['POST'])
def virtual_fitting():
    """
    Perform virtual try-on using the selected person and cloth images.
    """
    data = request.get_json()
    if not data or 'person_image' not in data or 'cloth_image' not in data:
        return jsonify({"error": "Missing image paths"}), 400

    person_image_rel_path = data['person_image'].lstrip('/')
    cloth_image_rel_path = data['cloth_image'].lstrip('/')

    # Construct absolute paths
    person_image_abs_path = os.path.join(PROJECT_ROOT, person_image_rel_path)
    cloth_image_abs_path = os.path.join(PROJECT_ROOT, cloth_image_rel_path)

    if not os.path.exists(person_image_abs_path):
        return jsonify({"error": f"Person image not found: {person_image_abs_path}"}), 404
    if not os.path.exists(cloth_image_abs_path):
        return jsonify({"error": f"Cloth image not found: {cloth_image_abs_path}"}), 404

    # Determine cloth type from path
    if 'upper' in cloth_image_rel_path:
        cloth_type = 'upper'
    elif 'overall' in cloth_image_rel_path:
        cloth_type = 'overall'
    else:
        # Default or error
        return jsonify({"error": "Could not determine cloth type from path"}), 400

    try:
        # --- 1. Generate Agnostic Mask ---
        print("Generating agnostic mask...")
        mask_info = automasker(person_image_abs_path, cloth_type)
        mask = mask_info['mask']

        # --- 2. Preprocess Images ---
        print("Preprocessing images...")
        width, height = 384, 512
        person_image = Image.open(person_image_abs_path).convert("RGB")
        cloth_image = Image.open(cloth_image_abs_path).convert("RGB")

        person_tensor = vae_processor.preprocess(person_image, height, width)[0]
        cloth_tensor = vae_processor.preprocess(cloth_image, height, width)[0]
        mask_tensor = mask_processor.preprocess(mask, height, width)[0]
        
        # Add batch dimension and move to device
        person_tensor = person_tensor.unsqueeze(0).to(device, dtype=torch.float16)
        cloth_tensor = cloth_tensor.unsqueeze(0).to(device, dtype=torch.float16)
        mask_tensor = mask_tensor.unsqueeze(0).to(device, dtype=torch.float16)

        # --- 3. Run Inference ---

        # --- TWEAKABLE PARAMETER ---
        # Try changing this value. Higher values make the model follow the cloth image more strictly.
        # Good values to try are 3.0, 3.5, 4.0. Too high might degrade quality.
        guidance_scale = 2.5 
        # ---------------------------

        print("Running inference...")
        generator = torch.Generator(device=device).manual_seed(555)
        with torch.no_grad():
            result_tensor = pipeline(
                person_tensor,
                cloth_tensor,
                mask_tensor,
                num_inference_steps=50,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )

        # --- 4. Post-process and Save ---
        print("Saving result...")
        result_image = to_pil_image(result_tensor)[0]
        
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
            
        person_name = os.path.splitext(os.path.basename(person_image_rel_path))[0]
        cloth_name = os.path.splitext(os.path.basename(cloth_image_rel_path))[0]
        timestamp = int(time.time())
        result_filename = f"{person_name}_{cloth_name}_{timestamp}.png"
        result_abs_path = os.path.join(RESULT_DIR, result_filename)
        
        result_image.save(result_abs_path)
        
        result_web_path = f"/images/result/{result_filename}"
        print(f"Result saved to: {result_abs_path}")

        return jsonify({"result_image_url": result_web_path})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred during fitting."}, 500)


if __name__ == '__main__':
    print("Starting CatVTON web application...")
    print(f"Models are loaded on device: {device}")
    print("Open http://127.0.0.1:5000/ffiting.html in your browser to access the application.")
    app.run(host='127.0.0.1', port=5000, debug=False)
