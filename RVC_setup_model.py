import os
import subprocess
import csv
import torch
import shutil
import sys
import requests
from pathlib import Path
import threading
from config import DATA_DIR, TRAINED_MODEL_DIR, LOGS_DIR
from model_setup.setup_utils import install_dependencies
from model_setup.file_utils import move_files
from model_setup.dataset_utils import sanitize_directory
from model_setup.model_utils import download_pretrain
from model_setup.train_utils import train_model

# Install the necessary dependencies
print("--------------------------------------")
print("|             RVC SETUP              |")
print("--------------------------------------")
choice = input("Do you want to install dependencies? (y/n): ")
if choice.lower() == 'y':
    install_dependencies()

# Set up the dataset
os.system('cls')
print("Welcome to the RVC Setup!") 
print("---------------------------------")

# Step 1: Experiment Name
experiment_name = input("Enter the experiment name: ")

# Step 2: Path to Training Folder
print("\nStep 2: Path to Training Folder")
print("WARNING!!! add slash in the end")
path_to_training_folder = input("Enter the path to the training folder: ")
print(path_to_training_folder)

# Step 3: Model Architecture
print("\nStep 3: Model Architecture")
print("Choose a model architecture (v1/v2): ")
model_architecture = input("Enter your choice: ")

# Step 4: Target Sample Rate
print("\nStep 4: Target Sample Rate")
print("Choose a target sample rate (32k/40k/48k): ")
target_sample_rate = input("Enter your choice: ")

# Step 5: Speaker ID
print("\nStep 5: Speaker ID")
print("Enter the speaker ID (0 for multi-speaker): ")
speaker_id = input("Enter your choice: ")

# Step 6: Pitch Extraction Algorithm
print("\nStep 6: Pitch Extraction Algorithm")
print("Choose a pitch extraction algorithm (harvest/crepe/rmvpe): ")
pitch_extraction_algorithm = input("Enter your choice: ")
print("Enter the crepe hop length (default is 64): ")
crepe_hop_length = input("Enter your choice: ")

# Step 7: Pretrain Type
print("\nStep 7: Pretrain Type")
print("Choose a pretrain type (original/OV2Super/RIN_E3/ItaIla/SnowieV3/SnowieV3xRIN_E3): ")
pretrain_type = input("Enter your choice: ")

ngpu = torch.cuda.device_count()
if_gpu_ok = False
if torch.cuda.is_available() or ngpu != 0:
    gpu_infos = []
    mem = []
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ["10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", "70", "80", "90", "M4", "T4", "TITAN"]):
            if_gpu_ok = True
            print("Compatible GPU detected: %s" % gpu_name)
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
else:
    raise Exception("No GPU detected; training cannot continue.")
gpus = "-".join(str(i[0]) for i in gpu_infos)

cpu_threads = os.cpu_count()

# Move the Mangio-RVC-Fork directory into the path_to_training_folder
move_files(f"{path_to_training_folder}/Mangio-RVC-Fork", path_to_training_folder)

# Download the pre-trained model
download_pretrain(pretrain_type, target_sample_rate, model_architecture)

# Setup the experiment
print("Setting up...")
expDir = os.path.join(path_to_training_folder, 'dataset', experiment_name)
os.makedirs(expDir, exist_ok=True)

if not os.path.isdir("csvdb/"):
    os.makedirs("csvdb")
    frmnt, stp = open("csvdb/formanting.csv", "w", newline=""), open("csvdb/stop.csv", "w", newline="")
    csv_writer = csv.writer(frmnt, delimiter=",")
    csv_writer.writerow([False, 1.0, 1.0])
    csv_writer = csv.writer(stp, delimiter=",")
    csv_writer.writerow([False])
    frmnt.close()
    stp.close()

DoFormant, Quefrency, Timbre = False, 1.0, 1.0

directories = []

final_directory = './dataset'
temp_directory = './temp_dataset'

if os.path.exists(final_directory):
    print("Dataset folder already found. Wiping...")
    shutil.rmtree(final_directory)
if os.path.exists(temp_directory):
    print("Temporary folder already found. Wiping...")
    shutil.rmtree(temp_directory)

os.makedirs(final_directory, exist_ok=True)
os.makedirs(temp_directory, exist_ok=True)

dataset_path = input("Enter the path to the dataset: ")

subprocess.check_call(['tar', '-xvf', dataset_path, '-C', temp_directory])
print("Sanitizing...")
sanitize_directory(temp_directory)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

directories = []  
if len(directories) == 0:
    print("Dataset Type: Audio Files (Single Speaker)")
    expDir = os.path.join(final_directory, experiment_name)
    os.makedirs(expDir, exist_ok=True)
    for r, _, f in os.walk(temp_directory):
        for name in f:
            shutil.copy2(f'{temp_directory}/dataset/{name}', expDir)
elif len(directories) == 1:
    print("Dataset Type: Single Speaker")
    fi = os.path.join(temp_directory, experiment_name)
    os.rename(directories[0], fi)
    shutil.move(fi, final_directory)
else:
    print("Dataset Type: Multispeaker")
    for fi in directories:
        shutil.move(fi, final_directory)

shutil.rmtree(temp_directory)

print("Dataset imported.")

assert cpu_threads > 0, "CPU threads not allocated correctly."

sr = int(target_sample_rate.rstrip('k')) * 1000
pttf = os.path.join(path_to_training_folder, "dataset", experiment_name)
exp_dir = os.path.join("logs", experiment_name)
os.makedirs(exp_dir, exist_ok=True)

cmd = f"CUDA_VISIBLE_DEVICES={gpus} python trainset_preprocess_pipeline_print.py \"{pttf}\" {sr} {cpu_threads} \"{exp_dir}\" 1"
subprocess.check_call(['python', 'trainset_preprocess_pipeline_print.py', f'{pttf}', f'{sr}', f'{cpu_threads}', f'{exp_dir}', '1'])

cmd = f"CUDA_VISIBLE_DEVICES={gpus} python extract_f0_print.py \"{exp_dir}\" {cpu_threads} {pitch_extraction_algorithm} {crepe_hop_length}"
subprocess.check_call(['python', 'extract_f0_print.py', f'{exp_dir}', f'{cpu_threads}', f'{pitch_extraction_algorithm}', f'{crepe_hop_length}'])

leng = len(gpus)
cmd = f"CUDA_VISIBLE_DEVICES={gpus} python extract_feature_print.py \"{model_architecture}\" {leng} 0 0 \"{exp_dir}\" {model_architecture}"
subprocess.check_call(['python', 'extract_feature_print.py', f'{model_architecture}', f'{leng}', '0', '0', f'{exp_dir}', f'{model_architecture}'])

# Train the model
train_model(experiment_name, path_to_training_folder, target_sample_rate, model_architecture, pitch_extraction_algorithm, crepe_hop_length, pretrain_type)

print("Done!")