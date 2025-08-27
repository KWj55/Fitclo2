import os
import json

# --- Configuration ---
# The root directory of the project.
# The script assumes it's run from c:\Project\CatVTON-edited\
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directories to scan for images, relative to PROJECT_ROOT
IMAGE_DIRS_TO_SCAN = [
    "images/human/men",
    "images/human/women",
]

# Output JavaScript file path, relative to PROJECT_ROOT
OUTPUT_JS_FILE = os.path.join("ffiting_html", "image_paths.js")

# Variable name to be used in the JavaScript file
JS_VARIABLE_NAME = "PERSON_IMAGE_PATHS"
# --- End of Configuration ---

def scan_images(base_dir, relative_dirs):
    """Scans given directories for image files and returns their relative paths."""
    all_paths = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for rel_dir in relative_dirs:
        full_dir_path = os.path.join(base_dir, rel_dir)
        if not os.path.isdir(full_dir_path):
            print(f"Warning: Directory not found, skipping: {full_dir_path}")
            continue

        print(f"Scanning directory: {full_dir_path}")
        for filename in sorted(os.listdir(full_dir_path)):
            if os.path.splitext(filename)[1].lower() in supported_extensions:
                # Create a relative path from the HTML file's location
                js_relative_path = os.path.join("..", rel_dir, filename).replace("\\", "/")
                all_paths.append(js_relative_path)
    return all_paths

def write_js_file(paths, output_file_path, var_name):
    """Writes the list of paths to a JavaScript file."""
    json_array = json.dumps(paths, indent=4)
    js_content = f"const {var_name} = {json_array};\n"

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"\nSuccessfully generated {output_file_path} with {len(paths)} image paths.")

if __name__ == "__main__":
    print("--- Starting Image List Generator ---")
    image_paths = scan_images(PROJECT_ROOT, IMAGE_DIRS_TO_SCAN)
    output_path_full = os.path.join(PROJECT_ROOT, OUTPUT_JS_FILE)
    write_js_file(image_paths, output_path_full, JS_VARIABLE_NAME)
    print("--- Script finished ---")