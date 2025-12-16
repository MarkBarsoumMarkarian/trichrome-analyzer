import os
import shutil
import re

# --- Configuration ---
SOURCE_DIR = "File_path"

# 2. The folder where the extracted images will be copied.
DESTINATION_DIR = "Output_path"

# 3. List of common image file extensions (case-insensitive)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg')

# --- Main Logic ---

def generate_unique_filename(base_name, dest_folder):
    """
    Checks if a file exists in the destination folder. If it does, 
    appends a unique numerical suffix (e.g., (1), (2), etc.) before the extension.
    Returns the final, unique filename.
    """
    # Split the filename into name and extension (e.g., 'tripod' and '.jpg')
    name, ext = os.path.splitext(base_name)
    
    # Start with the original filename
    unique_name = base_name
    
    # Counter for duplicates
    counter = 0

    # Loop until a file with the unique name does not exist in the destination
    while os.path.exists(os.path.join(dest_folder, unique_name)):
        counter += 1
        # Recreate the filename with the new suffix: name(counter).ext
        unique_name = f"{name}({counter}){ext}"
    
    return unique_name


def extract_matching_images():
    """
    Recursively searches SOURCE_DIR for images:
    1. Must be a recognized image file.
    2. Must start with 'tri' or 'Tri'.
    3. **NEW: Must contain the '%' character.**
    Copies them to DESTINATION_DIR, and handles duplicates with (x) numbering.
    """
    print(f"Starting search in: {SOURCE_DIR}")
    print(f"Target destination: {DESTINATION_DIR}")

    # Create the destination directory if it doesn't exist
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    
    count = 0

    # os.walk efficiently traverses the entire folder structure
    for root, _, files in os.walk(SOURCE_DIR):
        for filename in files:
            
            # Check 1: Must be a recognized image file
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                
                # Check 2: Must start with 'tri' or 'Tri'
                if re.match(r"^tri", filename, re.IGNORECASE):
                    
                    # --- NEW CHECK ADDED HERE ---
                    # Check 3: Must contain the '%' character
                    if '%' in filename:
                        
                        source_path = os.path.join(root, filename)
                        
                        # Generate the unique name for the destination folder
                        new_filename = generate_unique_filename(filename, DESTINATION_DIR)
                        
                        destination_path = os.path.join(DESTINATION_DIR, new_filename) 
                        
                        try:
                            # shutil.copy2 copies the file along with metadata (timestamps)
                            shutil.copy2(source_path, destination_path)
                            
                            # Print the original name and the new, saved name
                            if new_filename != filename:
                                print(f"COPIED (Duplicate): '{filename}' -> '{new_filename}'")
                            else:
                                print(f"COPIED: '{filename}'")
                            
                            print(f"  Source folder: {os.path.relpath(root, SOURCE_DIR)}")
                            count += 1
                            
                        except Exception as e:
                            print(f"ERROR copying {filename}: {e}")

    print("\n--- Extraction Complete ---")
    if count > 0:
        print(f"Successfully extracted **{count}** images.")
        print(f"All images are located in: {DESTINATION_DIR}")
    else:
        print("No matching images were found.")

if __name__ == "__main__":

    extract_matching_images()
