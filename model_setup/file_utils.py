import os
import shutil

def move_files(source_dir, target_dir):
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)
        if os.path.isfile(source_file):
            shutil.move(source_file, target_file)

def sanitize_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            if filename == ".DS_Store" or filename.startswith("._") or not filename.endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a')):
                os.remove(file_path)
        elif os.path.isdir(file_path):
            if filename == "__MACOSX":
                shutil.rmtree(file_path)
                continue
            sanitize_directory(file_path)