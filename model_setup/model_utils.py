import os
import requests
import shutil

def download_pretrain(pretrain_type, target_sample_rate, model_architecture):
    if pretrain_type == "original" and model_architecture == "v2":
        g = f"https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0G{target_sample_rate}.pth"
        d = f"https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/f0D{target_sample_rate}.pth"
    if pretrain_type == "original" and model_architecture == "v1":
        g = f"https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/v1/f0G{target_sample_rate}.pth"
        d = f"https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/v1/f0D{target_sample_rate}.pth"
    if pretrain_type == "OV2Super" and target_sample_rate == "40k":
        g = f"https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super{target_sample_rate}G.pth"
        d = f"https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super{target_sample_rate}D.pth"
    if pretrain_type == "OV2Super" and target_sample_rate == "32k":
        g = f"https://huggingface.co/poiqazwsx/Ov2Super32kfix/resolve/main/f0Ov2Super32kD.pth"
        d = f"https://huggingface.co/poiqazwsx/Ov2Super32kfix/resolve/main/f0Ov2Super32kD.pth"
    if pretrain_type == "RIN_E3":
        g = "https://huggingface.co/MUSTAR/RIN_E3/resolve/main/RIN_E3_G.pth"
        d = "https://huggingface.co/MUSTAR/RIN_E3/resolve/main/RIN_E3_D.pth"
    if pretrain_type == "SnowieV3":
        g = f"https://huggingface.co/MUSTAR/SnowieV3.1-{target_sample_rate}/resolve/main/G_SnowieV3.1_{target_sample_rate}.pth"
        d = f"https://huggingface.co/MUSTAR/SnowieV3.1-{target_sample_rate}/resolve/main/D_SnowieV3.1_{target_sample_rate}.pth"
    if pretrain_type == "ItaIla":
        g = "https://huggingface.co/TheStinger/itaila/resolve/main/ItaIla_32k_G.pth"
        d = "https://huggingface.co/TheStinger/itaila/resolve/main/ItaIla_32k_D.pth"
    if pretrain_type == "SnowieV3xRIN_E3":
        g = "https://huggingface.co/MUSTAR/SnowieV3.1-X-RinE3-40K/resolve/main/G_Snowie-X-Rin_40k.pth"
        d = "https://huggingface.co/MUSTAR/SnowieV3.1-X-RinE3-40K/resolve/main/D_Snowie-X-Rin_40k.pth"
    if pretrain_type == "TITAN":
        g = f"https://huggingface.co/blaise-tk/TITAN/resolve/main/models/medium/{target_sample_rate}/pretrained/G-f0{target_sample_rate}-TITAN-Medium.pth"
        d = f"https://huggingface.co/blaise-tk/TITAN/resolve/main/models/medium/{target_sample_rate}/pretrained/D-f0{target_sample_rate}-TITAN-Medium.pth"

    pretrained_base = "pretrained/" if model_architecture == "v1" else "pretrained_v2/"
    unpt = f"_{pretrain_type}" if pretrain_type != "original" else ""

    print("Downloading your pretrained model...")
    response = requests.get(g, allow_redirects=True)
    with open (os.path.join('Mangio-RVC-Fork', pretrained_base) + 'f0G' + target_sample_rate + unpt + '.pth', 'wb') as file:
        file.write(response.content)
    response = requests.get(d, allow_redirects=True)
    with open (os.path.join('Mangio-RVC-Fork', pretrained_base) + 'f0D' + target_sample_rate + unpt + '.pth', 'wb') as file:
        file.write(response.content)

    print("Pretrain downloaded. Best of luck training!")