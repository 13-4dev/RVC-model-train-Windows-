

# RVC Train (Windows)

This project is based on the [Mangio RVC Fork](https://github.com/Mangio621/Mangio-RVC-Fork) and this [Google Colab notepad](https://colab.research.google.com/drive/1XIPCP9ken63S7M6b5ui1b36Cs17sP-NS#scrollTo=g3fR68Yfkayg).

## Requirements

- Python 3.10.12
- A GPU with CUDA cores (the exact amount of VRAM needed may vary)

## Setup

1. Install the dependencies from `requirements.txt`.
2. Run `RVC_train.bat`.
3. Start the setup process.
4. Download the extra files.
5. Specify the experiment name.
6. Provide the path to the directory where the repository is stored.
7. Choose the model architecture.
8. Set the sample rate.
9. Specify the speaker ID.
10. Choose a pitch extraction algorithm.
11. Select the pretrain type.

Once these steps are completed, a pretrained RVC model will be set up, and you can begin training.

## Model Training

1. Start the training process.
2. Specify the experiment name.
3. Choose the model architecture.
4. Select the pre-trained model type.
5. Set the sample rate.
6. Set the training frequency (e.g., `10`).
7. Set the total number of epochs (e.g., `500`).
8. Set the batch size (e.g., `8`).
9. Choose whether to save only the latest checkpoint (recommended if the process may be interrupted).
10. Decide whether to cache all training sets.
11. Choose whether to save a small final model.

After training is complete, the model will be saved in the `models + experiment_name` folder.

## Special Thanks

A huge thank you to the RVC development team for their efforts!
