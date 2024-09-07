import os
import shutil
import math
import numpy as np                             
from sklearn.cluster import MiniBatchKMeans         
import subprocess
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set flags and global variables
save_extra_files_to_local = False
force_mbkm = False
opt = []  # INITIALIZE opt AS EMPTY LIST

# Gather inputs once
experiment_name = input("Enter the experiment name: ").strip()
model_architecture = input("Enter the model architecture (v1 or v2): ").strip().lower()
pretrain_type_options = ['original', 'custom']
pretrain_type = input(f"Choose pretrained model type {pretrain_type_options}: ").strip().lower()
sample_rate_options = ['32k', '40k', '48k']
target_sample_rate = input(f"Choose sample rate {sample_rate_options}: ").strip().lower()
speaker_id = input("Enter speaker ID: ").strip()
save_frequency = int(input("Enter save frequency (e.g., 10): "))
total_epochs = int(input("Enter total epochs (e.g., 500): "))
batch_size = int(input("Enter batch size (e.g., 8): "))
save_only_latest_ckpt = input("Save only latest checkpoint? (True/False): ").strip().lower() == "true"
cache_all_training_sets = input("Cache all training sets? (True/False): ").strip().lower() == "true"
save_small_final_model = input("Save small final model? (True/False): ").strip().lower() == "true"

# Define directories and check paths
exp_dir = f"{os.getcwd()}\\logs\\{experiment_name}"
os.makedirs(exp_dir, exist_ok=True)

feature_dir = f"{exp_dir}\\3_feature256" if model_architecture == "v1" else f"{exp_dir}\\3_feature768"
if not os.path.exists(feature_dir):
    raise Exception("No features exist for this model yet.")

# Load features and shuffle
listdir_res = list(os.listdir(feature_dir))
if len(listdir_res) == 0:
    raise Exception("No features exist for this model yet.")

infos = []
npys = []
for name in sorted(listdir_res):
    phone = np.load(f"{feature_dir}\\{name}")
    npys.append(phone)
big_npy = np.concatenate(npys, 0)
big_npy_idx = np.arange(big_npy.shape[0])
np.random.shuffle(big_npy_idx)

# Apply clustering if necessary
if big_npy.shape[0] > 2e5 or force_mbkm:
    big_npy = MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256, compute_labels=False, init="random").fit(big_npy).cluster_centers_

np.save(f"{exp_dir}\\total_fea.npy", big_npy)

# Create index and add data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
big_npy_tensor = torch.tensor(big_npy).float().to(device)
n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
k = 1
data_dim = big_npy.shape[1]
index = nn.Linear(data_dim, n_ivf, bias=False).to(device)

rand_vectors = torch.zeros((1, data_dim), dtype=torch.float32, device=device)
rand_vectors.bernoulli_(0.5)

for i in range(n_ivf):
    index.weight[i, :] = rand_vectors[i%len(rand_vectors)].clone()
    
query_input = torch.randn(1, data_dim, device=device)
distance = F.pairwise_distance(query_input, big_npy_tensor)
_, idx = torch.topk(distance, k=k, largest=False, dim=1)
index.query = idx.flatten()
torch.save(index, f"{exp_dir}\\trained_IVF{n_ivf}_Flat_nprobe_{k}_{experiment_name}_{model_architecture}.index")
print(f"Saved index at: {exp_dir}\\trained_IVF{n_ivf}_Flat_nprobe_{k}_{experiment_name}_{model_architecture}.index")

# Handle local file saving if flag is set
if save_extra_files_to_local:
    shutil.copy(f"{exp_dir}\\total_fea.npy", exp_dir)
    shutil.copy(f"{exp_dir}\\trained_IVF{n_ivf}_Flat_nprobe_{k}_{experiment_name}_{model_architecture}.index", exp_dir)

# Ensure pretrained model paths are correct
pretrained_base = "pretrained\\" if model_architecture == "v1" else "pretrained_v2\\"
unpt = f"_{pretrain_type}" if pretrain_type != "original" else ""
pretrainedD = os.path.join(pretrained_base, f"f0D{target_sample_rate}{unpt}.pth")
pretrainedG = os.path.join(pretrained_base, f"f0G{target_sample_rate}{unpt}.pth")

# Determine log interval
log_interval = 1
liFolderPath = os.path.join(exp_dir, "1_16k_wavs")
if os.path.exists(liFolderPath) and os.path.isdir(liFolderPath):
    wav_files = [f for f in os.listdir(liFolderPath) if f.endswith(".wav")]
    if wav_files:
        sample_size = len(wav_files)
        log_interval = math.ceil(sample_size / batch_size)
        if log_interval > 1:
            log_interval += 1
if log_interval > 250:
    log_interval = 200

# Build and log the command
cmd = (
    f"python train_nsf_sim_cache_sid_load_pretrain.py -e \"{experiment_name}\" "
    f"-sr {target_sample_rate} -f0 1 -bs {batch_size} -g 0 -te {total_epochs} -se {save_frequency} "
    f"{'-pg ' + pretrainedG if pretrainedG else ''} {'-pd ' + pretrainedD if pretrainedD else ''} "
    f"-l {1 if save_only_latest_ckpt else 0} -c {1 if cache_all_training_sets else 0} "
    f"-sw {1 if save_small_final_model else 0} -v {model_architecture} -li {log_interval}"
)

# Prepare the file list for training
gt_wavs_dir = f"{exp_dir}\\0_gt_wavs"
f0_dir = f"{exp_dir}\\2a_f0"
f0nsf_dir = f"{exp_dir}\\2b-f0nsf"

for dir_path in [gt_wavs_dir, feature_dir, f0_dir, f0nsf_dir]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

names = (
    set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    & set(name.split(".")[0] for name in os.listdir(feature_dir))
    & set(name.split(".")[0] for name in os.listdir(f0_dir))
    & set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
)

opt = [
    f"{gt_wavs_dir}\\{name}.wav|{feature_dir}\\{name}.npy|{f0_dir}\\{name}.wav.npy|{f0nsf_dir}\\{name}.wav.npy|{speaker_id}"
    for name in names
]

fea_dim = 256 if model_architecture == "v1" else 768
for _ in range(2):
    opt.append(
        f"{exp_dir}\\mute\\0_gt_wavs\\mute{target_sample_rate}.wav|{exp_dir}\\mute\\3_feature{fea_dim}\\mute.npy|{exp_dir}\\mute\\2a_f0\\mute.wav.npy|{exp_dir}\\mute\\2b-f0nsf\\mute.wav.npy|{speaker_id}"
    )

shuffle(opt)
filelist_path = f"{exp_dir}\\filelist.txt"
with open(filelist_path, "w") as f:
    f.write("\n".join(opt))
print(f"Filelist written to {filelist_path}.")

# Move mute files if they exist
source = '.\\logs\\mute'
destination = f'.\\logs\\{experiment_name}\\mute'
if os.path.exists(destination):
    print(f"{destination} already exists.")
else:
    shutil.move(source, destination)
    print(f'{source} --> {destination}')

MODEL_EXPORT_PATH = ".\\models\\" + experiment_name
MODEL_SOURCE_PATH = ".\\logs\\" + experiment_name

if not os.path.exists(MODEL_EXPORT_PATH):
    os.makedirs(MODEL_EXPORT_PATH)

skip_models = True
manual_save = True
STEPOUNT = 000
EPOCHCOUNT = 000

finished = False
potential = ".\\weights\\" + experiment_name + ".pth"
if os.path.exists(potential):
    finished = True

print("Detecting latest model...")
if not manual_save:
    currentMax = 0
    for r, _, f in os.walk(".\\weights\\"):
        for name in f:
            if name.endswith(".pth") and name != experiment_name + ".pth":
                if name.find(experiment_name) == -1:
                    continue
                pot = name.split('_')
                ep = pot[-2][1:]
                if not ep.isdecimal():
                    continue
                ep = int(ep)
                if ep > currentMax:
                    currentMax = ep
                    step = pot[-1].split('.')
                    step = int(step[0][1:])
                    EPOCHCOUNT = ep
                    STEPOUNT = step

TSTEP = STEPOUNT
if not skip_models:
    print("Copying model files...")
    if save_only_latest_ckpt:
        TSTEP = 2333333
    subprocess.run(f"copy {MODEL_SOURCE_PATH}\\D_{TSTEP}.pth {MODEL_EXPORT_PATH}", shell=True)
    subprocess.run(f"copy {MODEL_SOURCE_PATH}\\G_{TSTEP}.pth {MODEL_EXPORT_PATH}", shell=True)
    subprocess.run(f"copy {MODEL_SOURCE_PATH}\\config.json {MODEL_EXPORT_PATH}", shell=True)

print("Copying Tensorboard TFEVENT files...")
for r, d, f in os.walk(MODEL_SOURCE_PATH):
    for name in f:
        if name.startswith("events.out.tfevents") and os.path.exists(os.path.join(MODEL_SOURCE_PATH, name)):
            subprocess.run(f"copy {MODEL_SOURCE_PATH}\\{name} {MODEL_EXPORT_PATH}", shell=True)

print("Copying index files...")
for r, d, f in os.walk(MODEL_SOURCE_PATH):
    for name in f:
        if "index" in name:
            subprocess.run(f"copy {os.path.join(MODEL_SOURCE_PATH, name)} {MODEL_EXPORT_PATH}", shell=True)

print("Copying weight file...")
if finished:
    subprocess.run(f"copy {potential} {MODEL_EXPORT_PATH}", shell=True)
else:
    subprocess.run(f"copy \\\\content\\\\weights\\\\{experiment_name}_e{EPOCHCOUNT}_s{STEPOUNT}.pth {MODEL_EXPORT_PATH}", shell=True)

print("All done!")

print(f"Done! Model save in {MODEL_EXPORT_PATH}")