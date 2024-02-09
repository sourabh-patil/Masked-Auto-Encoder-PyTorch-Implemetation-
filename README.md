# Masked-Auto-Encoder-PyTorch-Implemetation

This is an easy-to-understand PyTorch Implementation of Masekd Auto Encoder. Link to the paper: https://arxiv.org/pdf/2111.06377.pdf

I have borrowed the model architecture from Facebook Research's official implementation github repository. I wanted to see how this model performs with a lightweight version and relatively smaller dataset. The model is trained on the Stanford Cars dataset which is free to download. I have added all the python scripts from dataloader, uitls to main training script and inference script. It was trained on a single Nvidia 2080 GPU for 6000 epochs using mixed precision. All the training details are in config.py and main.py 

![mae_block](https://github.com/sourabh-patil/Masked-Auto-Encoder-PyTorch-Implemetation-/assets/53788836/1505e718-7e14-43a0-9677-de73bf0b6fe5)

Results:

![results_mae](https://github.com/sourabh-patil/Masked-Auto-Encoder-PyTorch-Implemetation-/assets/53788836/7613f7b1-eb94-4863-b8bd-8ed2c08ccc44)
