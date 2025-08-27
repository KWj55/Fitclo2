import os
from flask import Flask, send_from_directory, jsonify, request

# --- Configuration ---
# Get the absolute path of the directory where the script is located
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Define paths to the HTML and image directories
FFITING_HTML_DIR = os.path.join(PROJECT_ROOT, 'ffiting_html')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')

# --- Flask App Setup ---
app = Flask(__name__)

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
    This replaces the need for the generate_image_list.py script.
    """
    image_type = request.args.get('type', 'person')
    
    image_paths = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    scan_dirs = []
    if image_type == 'person':
        # Directories to scan, relative to the IMAGES_DIR
        scan_dirs = ["human/men", "human/women"]
    elif image_type == 'clothing':
        # Directories to scan for clothing
        scan_dirs = ["cloth/overall", "cloth/upper"]
    
    for rel_dir in scan_dirs:
        full_dir_path = os.path.join(IMAGES_DIR, rel_dir)
        if not os.path.isdir(full_dir_path):
            print(f"Warning: Directory not found, skipping: {full_dir_path}")
            continue
        
        for filename in sorted(os.listdir(full_dir_path)):
            if os.path.splitext(filename)[1].lower() in supported_extensions:
                # Create a web-accessible path.
                # e.g., /images/human/men/000006_00.jpg
                web_path = os.path.join('images', rel_dir, filename).replace("\\", "/")
                image_paths.append(f'/{web_path}')
                
    return jsonify(image_paths)

if __name__ == '__main__':
    print("Starting CatVTON web application...")
    print("Open http://127.0.0.1:5000/ in your browser to access the application.")
    app.run(host='127.0.0.1', port=5000, debug=False)