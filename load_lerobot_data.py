"""Inspect the ForceVLA dataset stored in LeRobot format.

Usage:
    export HF_LEROBOT_HOME="$HOME/data/lerobot"
    python load_lerobot_data.py

The dataset will be downloaded automatically on first use if not already present
in the HF_LEROBOT_HOME directory.
"""

from pprint import pprint

import torch

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# ForceVLA training dataset hosted on HuggingFace.
# See: https://huggingface.co/datasets/qiaojunyu/ForceVLA-real-data
repo_id = "qiaojunyu/ForceVLA-real-data"

# Fetch lightweight metadata first (no large data download).
ds_meta = LeRobotDatasetMetadata(repo_id)

print(f"Total number of episodes: {ds_meta.total_episodes}")
print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
print(f"Frames per second used during data collection: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

# Load the full dataset (downloads data files on first use).
dataset = LeRobotDataset(repo_id)

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=32,
    shuffle=True,
)
camera_keys = dataset.meta.camera_keys

for batch in dataloader:
    for camera_key in camera_keys:
        print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
    print(f"{batch['observation.state'].shape=}")  # (32, state_dim)
    print(f"{batch['action'].shape=}")  # (32, action_dim)
    break
