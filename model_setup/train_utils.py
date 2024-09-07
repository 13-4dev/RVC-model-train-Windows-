import os
import csv
import torch
import shutil
import subprocess

def train_model(experiment_name, path_to_training_folder, target_sample_rate, model_architecture, pitch_extraction_algorithm, crepe_hop_length, pretrain_type):
    # Set up necessary paths
    now_dir = os.getcwd()
    loc = "%s/logs/%s" % (now_dir, experiment_name)

    # Set up the CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Copy the logs to a zip file
    subprocess.check_call(['powershell', '-Command', f'Compress-Archive -Path {loc} -DestinationPath rvcLogs.zip -Force'])

    # Move the Mangio-RVC-Fork directory to the parent directory
    source_folder = './Mangio-RVC-Fork'
    destination_folder = os.getcwd()
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        destination_item = os.path.join(destination_folder, item)
        if os.path.isdir(source_item):
            shutil.move(source_item, destination_item)
        else:
            shutil.move(source_item, destination_folder)

    print("Training complete!")