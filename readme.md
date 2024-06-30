# Master Thesis 

This repository will hold all code and tex files for my master thesis at the LMU Munich.

## Install requirements for Parfam and GinnLP
```bash
git submodule foreach pip install -r requirements.txt
```

## Setup Cuda 
(Might already be satisfied through requirements.txt) \
Use https://pytorch.org/get-started/locally/ to install required packages, e.g.
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```