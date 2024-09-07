import os
import subprocess
import sys
from pathlib import Path

def update_pip():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip==23.2.1'])
    except subprocess.CalledProcessError as e:
        print(f"Error updating pip: {e}")

def execute_commands(commands):
    for command in commands:
        try:
            subprocess.check_call(command, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command {command}: {e}")
            break

def install_dependencies():
    update_pip()
    
    commands = [
        'pip install omegaconf==2.1.1',
        'python -m pip install setuptools wheel fairseq ffmpeg-python pyworld numpy==1.23.5 numba==0.56.4 librosa==0.9.2 ffmpeg praat-parselmouth pyworld',
        'git clone https://github.com/alexlnkp/Mangio-RVC-Tweaks.git',
        'robocopy Mangio-RVC-Tweaks Mangio-RVC-Fork /S /MOVE',
        'git clone https://github.com/maxrmorrison/torchcrepe.git',
        'robocopy torchcrepe\\torchcrepe Mangio-RVC-Fork\\ /S /MOVE',
        'rmdir /s /q torchcrepe',
        'powershell.exe -Command "Invoke-WebRequest -Uri https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/32k.json -OutFile Mangio-RVC-Fork\\configs\\32k.json"',
        'powershell.exe -Command "Invoke-WebRequest -Uri https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/40k.json -OutFile Mangio-RVC-Fork\\configs\\40k.json"',
        'powershell.exe -Command "Invoke-WebRequest -Uri https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/48k.json -OutFile Mangio-RVC-Fork\\configs\\48k.json"'
       ]
    
    execute_commands(commands)

    print("Installation complete.")