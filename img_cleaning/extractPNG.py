import os
import shutil

def extract_jpg_from_dwi_folders(source_path, destination_path):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    
    # Counter for renaming files to avoid duplicates
    file_counter = 1
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(source_path):
        # Check if current directory is named "DWI"
        if os.path.basename(root) == "DWI":
            # Look for .jpg files in current DWI folder
            for file in files:
                if file.lower().endswith('.jpg'):
                    source_file = os.path.join(root, file)
                    # Create new filename with counter to avoid overwriting
                    new_filename = f"image_{file_counter}.jpg"
                    destination_file = os.path.join(destination_path, new_filename)
                    
                    try:
                        # Copy the file to destination
                        shutil.copy2(source_file, destination_file)
                        print(f"Copied: {file} -> {new_filename}")
                        file_counter += 1
                    except Exception as e:
                        print(f"Error copying {file}: {str(e)}")

def main():
    # Define the source and destination paths
    source_path = "../dataset/Dataset_MRI_Folder/Normal"
    destination_path = "../raw_imgs/normal"
    
    # Verify source path exists
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist")
        return
    
    print(f"Starting extraction from: {source_path}")
    print(f"Destination folder: {destination_path}")
    
    # Extract the jpg files
    extract_jpg_from_dwi_folders(source_path, destination_path)
    
    print(f"\nExtraction complete. Files saved in: {destination_path}")

if __name__ == "__main__":
    main()