import os
import shutil
import csv
import threading
import xml.etree.ElementTree as ET

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