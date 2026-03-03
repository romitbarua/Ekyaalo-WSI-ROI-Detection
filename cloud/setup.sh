#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/Bpfrederick/Ekyaalo-WSI-ROI-Detection.git"
BRANCH="rb-productionzation"
DRIVE_FOLDER_ID="1OdmAmI53EVXWIM_z50ReOr-x3EP9GwzU"
WORKDIR="$HOME/wsi"

# nvidia drivers (GCP helper handles kernel module + CUDA)
curl -fsSL https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py \
    -o /tmp/install_gpu_driver.py
sudo python3 /tmp/install_gpu_driver.py
nvidia-smi

sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip git > /dev/null

mkdir -p "$WORKDIR" && cd "$WORKDIR"
[ -d "Ekyaalo-WSI-ROI-Detection" ] || git clone --branch "$BRANCH" "$REPO"
cd Ekyaalo-WSI-ROI-Detection

python3 -m venv .venv && source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q -r requirements.txt

gdown --folder "$DRIVE_FOLDER_ID" -O data/raw/ --remaining-ok
python split_by_patient.py --output data/metadata/metadata_patient_split.csv

python -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')"
ls data/raw/AnnotatedDataSet/ > /dev/null && echo "data ok" || echo "WARNING: images missing"
