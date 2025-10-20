#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# determine platform from .env file
from pathlib import Path
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # __file__ doesn't exist (e.g., in Jupyter)
    BASE_DIR = Path.cwd()
print(f"Base directory: {BASE_DIR}")

try:
    with open(BASE_DIR / ".env", "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            if key == "PLATFORM":
                PLATFORM = value
    print(f"Running on platform: {PLATFORM}")
except FileNotFoundError:
    PLATFORM = "PC"
    print("No .env file found. Defaulting to PC platform.")

# %%
# configure global variables based on platform
import torch

if PLATFORM == "PC":
    VISION_DEVICE = "cpu" # not enough GPU memory on laptop for full vision model
    MAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # training with WikiArt hard labels
    PRELIM_TRAINING = False
    # training again with expert-annotated labels
    # already done in active loop but extra epochs can be helpful
    ANNOTATED_TRAINING = True 

elif PLATFORM == "IDAS":
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available on IDAS. Please restart with GPU access.")
    VISION_DEVICE = "cuda"
    MAIN_DEVICE = "cuda"
    PRELIM_TRAINING = False
    ANNOTATED_TRAINING = True
else:
    raise ValueError(f"Unknown PLATFORM: {PLATFORM}")

print(f"Vision device: {VISION_DEVICE}, Main device: {MAIN_DEVICE}, Prelim training: {PRELIM_TRAINING}, Annotated training: {ANNOTATED_TRAINING}")
IMGS_DIR = BASE_DIR / "paintings"
PRELIM_METADATA_PATH = BASE_DIR / "metadata" / "paintings_metadata_with_hard_labels.json"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# vector dimensions
MOVEMENT_DIM = 6
GENRE_DIM = 6
FORM_DIM = 6

# %%

# %%
# load BLIP2 model and processor
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration

model_name = "Salesforce/blip2-flan-t5-xl"
local_model_path =  BASE_DIR / "blip2_model"

if os.path.exists(local_model_path):
    print("Loading model from local directory...")
    processor = Blip2Processor.from_pretrained(local_model_path, use_fast=True)
    blip2 = Blip2ForConditionalGeneration.from_pretrained(local_model_path)
else:
    print("Downloading model from Hugging Face...")
    processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
    blip2 = Blip2ForConditionalGeneration.from_pretrained(model_name)

    # Save to local directory for future use
    processor.save_pretrained(local_model_path)
    blip2.save_pretrained(local_model_path)

print("model and processor loaded")
blip2.to(VISION_DEVICE)  # Load model on CPU first if on computer
print(f"model sent to {VISION_DEVICE}")

# Freeze vision encoder to save memory; we are not training the vision encoder
for param in blip2.vision_model.parameters():
    param.requires_grad = False


# %%
# checkpoint functions
import os
import glob

def get_latest_checkpoint():
    checkpoint_files = glob.glob(os.path.join(CHECKPOINTS_DIR, "model_*.pt"))
    if not checkpoint_files:
        return None
    # Sort by modification time
    checkpoint_files.sort(key=os.path.getmtime)
    return checkpoint_files[-1]

# as we are not training the vision model, load only the relevant parts
def load_model_from_latest(model):
    latest_check_point = get_latest_checkpoint()
    if latest_check_point is None:
        print("No checkpoint found. Starting from scratch.")
        return
    state_dict = torch.load(latest_check_point, map_location='cpu')
    
    model.shared_features.load_state_dict(state_dict["shared_features"])
    model.movement_head.load_state_dict(state_dict["movement_head"])
    model.genre_head.load_state_dict(state_dict["genre_head"])
    # the earlier name, changed for clarity
    if "style_head" in state_dict:
        model.form_head.load_state_dict(state_dict["style_head"])
    elif "form_head" in state_dict:
        model.form_head.load_state_dict(state_dict["form_head"])

    if "qformer" in state_dict:
        model.blip2.qformer.load_state_dict(state_dict["qformer"])
        print(" Loaded Q-Former weights")
    
    print(f" Loaded weights from {latest_check_point}")


def save_progress(model, file_name):
    checkpoint_file = os.path.join(CHECKPOINTS_DIR, f"{file_name}_{PLATFORM}.pt")

    state_dict = {
        "shared_features": model.shared_features.state_dict(),
        "movement_head": model.movement_head.state_dict(),
        "genre_head": model.genre_head.state_dict(),
        "form_head": model.form_head.state_dict(),
    }
    # Optionally include Q-Former if it's being trained
    if any(p.requires_grad for p in model.blip2.qformer.parameters()):
        state_dict["qformer"] = model.blip2.qformer.state_dict()
        
    torch.save(state_dict, checkpoint_file)
    print(f" Saved fine-tuned modules to: {checkpoint_file}")


# %%
# load/create persistent train/test split
import os
import json
import random
def get_split( valid_ids: list, test_percentage: float = 0.2, 
              split_file: str = BASE_DIR/"metadata"/"data_splits.json", seed: int = 42,):
    """
    Create or load a consistent train/test split for a dataset of a given size.
    A split is saved/loaded based on the total_length of the dataset.

    Args:
        total_length (int): Total number of samples/images.
        test_percentage (float): Fraction of samples to use for testing.
        split_file (str): JSON file path to store splits.
        seed (int): Random seed for reproducibility.

    Returns:
        (train_indices, test_indices): Two lists of indices.
    """
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            split_data = json.load(f)
    else:
        split_data = {}

    # Ensure keys exist
    train = split_data.get("train", [])
    test = split_data.get("test", [])
    if train and test:
        print(f"Loaded existing train/test split from {split_file}")
        return train, test


    # Otherwise, generate a new split
    random.seed(seed)
    random.shuffle(valid_ids)
    num_test = int(len(valid_ids) * test_percentage)
    test_ids = valid_ids[:num_test]
    train_ids = valid_ids[num_test:]

    split_data["train"] = train_ids
    split_data["test"] = test_ids
    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=2)

    return train_ids, test_ids



# %%
# dataloader creation for preliminary; training lazy loading is neeeded due to dataset size
# only necessary for preliminary training, as FastAPI backend provides experted annotated data
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

class PaintingDataset(Dataset):
    def __init__(self, image_paths, targets):
        self.image_paths = image_paths
        self.targets = targets
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        target = self.targets[idx]
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return path, target_tensor

# need a collate function because torch doesn't work well with lists of strings
def collate_fn(batch):
    paths, targets = zip(*batch)
    targets_tensor = torch.stack(targets)
    return list(paths), targets_tensor


def create_train_test_loaders( batch_size_train=32, batch_size_test=32, test_percentage=0.1):
    print("="*80, "LOADING DATASET", "="*80)
    all_files = sorted(os.listdir(IMGS_DIR))
    
    with open(PRELIM_METADATA_PATH, 'r', encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} paintings")

    valid_paths_dict = {}
    for file_name in all_files:
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(IMGS_DIR, file_name)
            # Extract ID from filename (first part before underscore)
            image_id = file_name.split("_")[0]
            if image_id in metadata and "rough_groundtruth" in metadata[image_id]:
                valid_paths_dict[image_id] = path

    print(f"Found {len(valid_paths_dict)} image files in {IMGS_DIR}")
    if len(valid_paths_dict) == 0:
        raise ValueError("No valid images found with corresponding metadata.")


    # split into train and test sets
    train_ids, test_ids = get_split(valid_ids=list(valid_paths_dict.keys()), test_percentage=test_percentage)


    train_paths = [valid_paths_dict[id] for id in train_ids]
    train_targets = [metadata[id]["rough_groundtruth"] for id in train_ids]

    test_paths = [valid_paths_dict[id] for id in test_ids]
    test_targets = [metadata[id]["rough_groundtruth"] for id in test_ids]
    print(f"\nTrain: {len(train_paths)} images, Test: {len(test_paths)} images")

    train_dataset = PaintingDataset(train_paths, train_targets)
    test_dataset = PaintingDataset(test_paths, test_targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    return train_loader, test_loader

if PRELIM_TRAINING:
    train_loader, test_loader = create_train_test_loaders(
        batch_size_train=32,
        batch_size_test=32,
        test_percentage=0.1,
    )


# %%

# %%
# GPU memory monitoring utility
def print_gpu_mem(prefix="GPU"):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2   # MB
        reserved = torch.cuda.memory_reserved() / 1024**2     # MB
        print(f"{prefix} Memory — Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    else:
        print("CUDA not available")


# %%
# Multi-head regression model
import torch
import torch.nn as nn

class BLIP2MultiHeadRegression(nn.Module):
    def __init__(self, blip2_model,
                 use_form_head=True,
                 train_qformer=False,
                 train_vision=False):
        super().__init__()

        # --- Core model ---
        self.blip2 = blip2_model
        self.use_form_head = use_form_head

        # --- Control what's trainable ---
        for param in self.blip2.vision_model.parameters():
            param.requires_grad = train_vision
        for param in self.blip2.qformer.parameters():
            param.requires_grad = train_qformer

        # --- Move modules to appropriate devices ---
        self.blip2.vision_model.to(VISION_DEVICE)
        self.blip2.qformer.to(MAIN_DEVICE)

        # query_tokens is an nn.Parameter → rewrap properly after moving
        self.blip2.query_tokens = nn.Parameter(
            self.blip2.query_tokens.to(MAIN_DEVICE)
        )

        # --- Config info ---
        num_query_tokens = blip2_model.config.num_query_tokens
        hidden_size = blip2_model.config.qformer_config.hidden_size
        feature_dim = num_query_tokens * hidden_size

        print(f"Num query tokens: {num_query_tokens}")
        print(f"Hidden size: {hidden_size}")
        print(f"Feature dim: {feature_dim}")
        print(f"Use form head: {use_form_head}")

        # --- Shared feature extraction ---
        self.shared_features = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(MAIN_DEVICE)

        # --- Movement and Genre heads ---
        self.movement_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, MOVEMENT_DIM)
        ).to(MAIN_DEVICE)

        self.genre_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, GENRE_DIM)
        ).to(MAIN_DEVICE)

        # --- Form head (always defined, but only used if enabled) ---
        self.form_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, FORM_DIM)
        ).to(MAIN_DEVICE)

    def forward(self, images):
        # --- Vision encoding ---
        images_vision = images.to(VISION_DEVICE)

        if self.training and next(self.blip2.vision_model.parameters()).requires_grad:
            vision_outputs = self.blip2.vision_model(pixel_values=images_vision)
        else:
            with torch.no_grad():
                vision_outputs = self.blip2.vision_model(pixel_values=images_vision)

        image_embeds = vision_outputs.last_hidden_state.to(MAIN_DEVICE)  # move to GPU

        # --- Q-Former processing ---
        query_tokens = self.blip2.query_tokens.expand(images.shape[0], -1, -1).to(MAIN_DEVICE)
        image_attention_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long).to(MAIN_DEVICE)

        query_outputs = self.blip2.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )

        # --- Flatten Q-Former output ---
        query_hidden_states = query_outputs.last_hidden_state
        flattened = query_hidden_states.flatten(start_dim=1)

        # --- Shared features ---
        shared_features = self.shared_features(flattened)

        # --- Regression heads ---
        movement_scores = self.movement_head(shared_features)
        genre_scores = self.genre_head(shared_features)
        form_scores = self.form_head(shared_features)

        outputs = {
            'movement': movement_scores,
            'genre': genre_scores,
            'form': form_scores,
        }
        return outputs



# %%
# Multi-head weighted loss function
class WeightedMultiHeadLoss(nn.Module):
    def __init__(self, movement_weight=1.0, genre_weight=1.0, form_weight=1.0, use_form=True):
        super().__init__()
        self.movement_weight = movement_weight
        self.genre_weight = genre_weight
        self.form_weight = form_weight
        self.zoom_movement_factor = 1.0
        self.zoom_genre_factor = 1.0
        self.zoom_form_factor = 1.0
        self.use_form = use_form

        # MSE loss for continuous targets
        self.mse = nn.MSELoss(reduction='none')
        
        self.form_components_weights = torch.tensor(
            [1.0] * FORM_DIM,
            dtype=torch.float32
        ).to(MAIN_DEVICE)
    
    def set_zoom_level(self, zoom_level):
        form_weight_factors = [
            0.0,  # balance can only be assessed at full image
            0.7,  # complexity. high zoom might miss overall complexity
            0.5,  # emotionality. high zoom might crop out the emotional content
            0.55,  # dynamic. high zoom might crop out the dynamic part
            0.9,  # naturalistic can be partially assessed at medium zoom
            0.9   # texture is easily assessed at high zoom
        ]
        # special case for original image
        if zoom_level == 0:
            self.form_components_weights = torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
                dtype=torch.float32
            ).to(MAIN_DEVICE)
            self.zoom_genre_factor = 1.0
            self.zoom_movement_factor = 1.0
        else:
            self.form_components_weights = torch.tensor(
                [f ** zoom_level for f in form_weight_factors],
                dtype=torch.float32
            ).to(MAIN_DEVICE)
            self.zoom_genre_factor = 1.0 *  (0.4 ** zoom_level) # pretty hard to see the genre at high zooms
            self.zoom_movement_factor = 1.0 * (0.7 ** zoom_level) # movement is still somewhat visible at medium zooms

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with 'movement', 'genre', 'form' (raw outputs)
            targets: tensor [batch, total_dim] (target values)
        """
        # Split targets by head dimensions
        movement_target = targets[:, :MOVEMENT_DIM]
        genre_target    = targets[:, MOVEMENT_DIM : MOVEMENT_DIM + GENRE_DIM]

        # --- Movement loss ---
        movement_loss = self.mse(predictions['movement'], movement_target)
        movement_loss = movement_loss.mean() * self.movement_weight * self.zoom_movement_factor


        # --- Genre loss ---
        genre_loss = self.mse(predictions['genre'], genre_target)
        genre_loss = genre_loss.mean() * self.genre_weight * self.zoom_genre_factor

        # --- Total loss ---
        total_loss = movement_loss + genre_loss
        loss_dict = {'movement': movement_loss.item(), 'genre': genre_loss.item()}

        if self.use_form:
            self.form_components_weights = self.form_components_weights.to(targets.device)
            form_target = targets[:, MOVEMENT_DIM + GENRE_DIM :]
            form_loss = self.mse(predictions['form'], form_target)
            form_loss = form_loss * self.form_components_weights.unsqueeze(0)
            form_loss = form_loss.mean() * self.form_weight * self.zoom_form_factor
            total_loss += form_loss
            loss_dict['form'] = form_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# %%
# augmentation function for preliminary training
from PIL import Image, ImageOps, UnidentifiedImageError, ImageFile
import numpy as np
import torch

# Do NOT allow truncated images - raise errors instead
ImageFile.LOAD_TRUNCATED_IMAGES = False

def augment_batch(image_paths, targets, processor):
    """
    Load images, create flipped versions, and return pixel values.
    
    Args:
        image_paths: List of file paths to images (length batch_size)
        targets: Tensor of shape [batch_size, 12]
        processor: BLIP2 processor
        
    Returns:
        pixel_values: Tensor of shape [batch_size*2, 3, H, W]
        doubled_targets: Tensor of shape [batch_size*2, 12]
    """
    images = []
    doubled_targets = []
    
    # Iterate by index since targets is a tensor
    for idx in range(len(image_paths)):
        img_path = image_paths[idx]
        target = targets[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            print(f"Skipping corrupted image: {img_path} ({e})")
            continue
        
        # Add original image
        images.append(img)
        doubled_targets.append(target)
        
        # Add horizontally flipped image
        flipped_img = ImageOps.mirror(img)
        images.append(flipped_img)
        doubled_targets.append(target)
    
    if len(images) == 0:
        return None, None
    
    # Process all images at once with processor
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    
    # Stack all target rows into [num_images, 12]
    targets_tensor = torch.stack(doubled_targets)
    
    return pixel_values, targets_tensor


# %%
# training and testing epoch functions
import time
def test_epoch(model, dataloader, criterion, device, processor):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for image_paths, targets in dataloader:
            # Augment the batch (same as training for consistency)
            pixel_values, targets_tensor = augment_batch(image_paths, targets, processor)
            
            pixel_values = pixel_values.to(device, non_blocking=True)
            targets_tensor = targets_tensor.to(device, non_blocking=True)
            
            predictions = model(pixel_values)
            loss, _ = criterion(predictions, targets_tensor)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Validation complete | Avg Loss: {avg_loss:.4f}")
    return avg_loss

def train_epoch(model, dataloader, optimizer, criterion, device, processor, val_loader=None):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    images_processed = 0
    for step, (image_paths, targets) in enumerate(dataloader):
        # print(f"step: {step}, image_paths: {image_paths}, targets: {targets}")
        # Augment the batch
        pixel_values, targets_tensor = augment_batch(image_paths, targets, processor)
        if pixel_values == None:
            continue
        
        pixel_values = pixel_values.to(device, non_blocking=True)
        targets_tensor = targets_tensor.to(device, non_blocking=True)


        optimizer.zero_grad()
        predictions = model(pixel_values)
        loss, loss_dict = criterion(predictions, targets_tensor)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        time_elapsed = time.time() - start_time

        images_processed += len(pixel_values)
        
        if step % 10 == 0:
            print(f"Step {step + 1}/{len(dataloader)} | Images: {images_processed} | Time: {time_elapsed:.2f}s | Loss: {loss.item():.4f}")
            # print_gpu_mem()
        
        # the data size is large enough that we want to validate more than once per epoch to prevent overfitting
        if val_loader is not None and (step + 1) % 100 == 0:
            val_loss = test_epoch(model, val_loader, criterion, device, processor)
            print(f" Validation Loss after {step + 1} steps: {val_loss:.4f}")
        
    num_batches = len(dataloader)
    total_images = num_batches * dataloader.batch_size * 2  # *2 for augmentation
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    
    print(f"Epoch complete | Avg Loss: {avg_loss:.4f} | Total images: {total_images} | Time: {epoch_time:.2f}s")
    
    return avg_loss

# %%
# preliminary training function
import torch, json
def run_preliminary_training(
    blip2,
    processor,
    train_loader,
    test_loader,
    num_epochs=20,
    early_stopping_patience=5
):
    print("="*60 + "PRELIMINARY TRAINING" + "="*60)

    # ---------------- Model setup ----------------
    model = BLIP2MultiHeadRegression(
        blip2,
        use_form_head=False,
        train_qformer=True,
        train_vision=False
    )
    load_model_from_latest(model)

    criterion = WeightedMultiHeadLoss(
        movement_weight=1.0,
        genre_weight=1.0,
        use_form=False
    ).to(MAIN_DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-6,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # ---------------- Training loop ----------------
    history = {"train_loss": [], "val_loss": [], "learning_rates": []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.2e}")
        history["learning_rates"].append(current_lr)
        print(f"{'='*60}")

        # --- Training ---
        train_loss = train_epoch(model, train_loader, optimizer, criterion, MAIN_DEVICE, processor)
        history["train_loss"].append(train_loss)

        # --- Validation ---
        val_loss = test_epoch(model, test_loader, criterion, MAIN_DEVICE, processor)
        history["val_loss"].append(val_loss)

        # --- Improvement tracking ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        # --- Early stopping ---
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

        # --- Scheduler update ---
        scheduler.step(val_loss)

        # --- Save progress ---
        file_name = f"model_epoch_{epoch}_valLoss_{val_loss:.4f}"
        save_progress(model, file_name)

    # ---------------- Wrap up ----------------
    print("\n" + "="*30, "TRAINING COMPLETE", "="*30)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)

    # Save history
    history_path = CHECKPOINTS_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    return history

if PRELIM_TRAINING:
    history = run_preliminary_training(
        blip2,
        processor,
        train_loader,
        test_loader,
    )

# %%
# multi-zoom augmentation function for annotated images
import math
import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def augment_annotated_image(image: Image.Image, processor, max_zoom: int = 9, show_debug=False, max_images = 10) -> dict:
    """
    Breaks an image into multiple zoom levels and crops.

    Zoom level concept:
        - Zoom 0: full image, resized to target_dim x target_dim
        - Zoom >=1: square crops, each crop covers 1/zoom of the short side of original image
          (no stretching, only resized to target_dim x target_dim for model input)

    Args:
        image: PIL.Image
        processor: processor with .image_processor.size (dict with 'height' and 'width')
        max_zoom: maximum zoom level (higher = finer detail)
        show_debug: if True, shows crops in a row for debugging

    Returns:
        dict: {zoom_level: [pillow images]}
    """

    augmented_images_by_zoom = {}

    # Step 0: verify processor target size
    target_size = processor.image_processor.size
    target_dim = target_size['height']
    if target_size['height'] != 224 or target_size['width'] != 224:
        raise ValueError(f"Unexpected processor size {target_size}, expected (224, 224)")

    W, H = image.size
    short_side = min(W, H)

    total_images = 0

    # --- Zoom 0: full image ---
    full_resized = image.resize((target_dim, target_dim), Image.LANCZOS)
    imgs = [full_resized, ImageOps.mirror(full_resized)]
    augmented_images_by_zoom[0] = imgs
    total_images += len(imgs)
    print(f"Zoom 0 (full image): 2 images created")

    # --- Zoom >=1: crops ---
    zoom_levels = np.geomspace(1, max_zoom, num=min(max_zoom, 6))
    print(f"Zoom levels >=1: {zoom_levels}")

    for z in zoom_levels:
        crop_size = int(short_side / z)  # square crop in original pixels

        # adaptive stride: more overlap for lower zooms
        stride_ratio = 0.5 if z == 1 else min(1.0, 0.5 + 0.05 * (z - 1))
        stride = max(1, int(crop_size * stride_ratio))

        x_steps = max(1, math.ceil((W - crop_size) / stride) + 1)
        y_steps = max(1, math.ceil((H - crop_size) / stride) + 1)

        augmented_images_by_zoom[z] = []
        images_per_zoom = 0

        for xi in range(x_steps):
            for yi in range(y_steps):
                left = xi * stride
                upper = yi * stride
                right = left + crop_size
                lower = upper + crop_size

                # adjust if exceeding bounds
                if right > W:
                    left -= (right - W)
                    right = W
                if lower > H:
                    upper -= (lower - H)
                    lower = H

                cropped_img = image.crop((left, upper, right, lower))
                cropped_resized = cropped_img.resize((target_dim, target_dim), Image.LANCZOS)
                imgs = [cropped_resized, ImageOps.mirror(cropped_resized)]

                if show_debug:
                    # show images in a row
                    fig, axes = plt.subplots(1, len(imgs), figsize=(len(imgs)*2, 2))
                    if len(imgs) == 1:
                        axes = [axes]
                    for ax, im, idx in zip(axes, imgs, range(len(imgs))):
                        ax.imshow(im)
                        ax.set_title(f"{'mirror' if idx else 'orig'}")
                        ax.axis("off")
                    plt.suptitle(f"Zoom {z:.2f} crop covers 1/{z:.2f} of short side ({xi},{yi})")
                    plt.show(block=False)
                    plt.pause(0.001)
                    plt.close()

                augmented_images_by_zoom[z].extend(imgs)
                images_per_zoom += len(imgs)

        total_images += images_per_zoom
        if total_images > max_images:
            print(f"Reached max images limit of {max_images}, stopping augmentation.")
            break
        print(f"Zoom {z:.2f}: crop covers 1/{z:.2f} of short side → {images_per_zoom} images created")

    print(f"Total images created (including zoom 0): {total_images}")
    if max_images is not None and total_images > max_images:
        print(f"Truncating to maximum of {max_images} images total.")
        count = 0
        truncated_dict = {        }
        truncated_dict[0] = augmented_images_by_zoom[0]
        count += len(augmented_images_by_zoom[0])
        
        for z in sorted([k for k in augmented_images_by_zoom.keys() if k != 0]):
            truncated_dict[z] = []
            for img in augmented_images_by_zoom[z]:
                if count < max_images:
                    truncated_dict[z].append(img)
                    count += 1
                else:
                    print(f"Reached max images limit of {max_images}, stopping augmentation.")
                    return truncated_dict

    return augmented_images_by_zoom


# %%
# web access functions
import torch
from transformers import Blip2Processor

# --- Global variables for lazy loading ---
_model, _processor = None, None

def initialize_model_for_webaccess():
    """
    Initialize the BLIP2 multi-head regression model and processor.
    Loads the latest checkpoint if available.
    """
    model = BLIP2MultiHeadRegression(
        blip2,
        use_form_head=True,
        train_qformer=True,
        train_vision=False
    )

    load_model_from_latest(model)
    processor = Blip2Processor.from_pretrained(local_model_path, use_fast=True)
    return model, processor

def get_model_and_processor():
    """
    Lazy-load the model and processor.
    """
    global _model, _processor
    if _model is None or _processor is None:
        _model, _processor = initialize_model_for_webaccess()
        print(f"Model and processor ready")
    return _model, _processor

def forward_images(images):
    model, processor = get_model_and_processor()
    model.eval()

    # Process all images as a batch
    inputs = processor(images=images, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model(inputs)

    # Move each head to CPU and convert to list
    embeddings = torch.cat([ outputs['movement'], outputs['genre'], outputs['form'] ], 
                           dim=1).cpu().tolist()
    


    print(f"Forward pass completed on {len(images)} images")
    return embeddings

def backward_single_image(image, target, lr=1e-5, batch_size=4,
                          movement_weight=1.0, genre_weight=1.0, form_weight=1.0,
                          FREEZE_QFORMER=False, FREEZE_SHARED=False,
                          image_id="unknown", max_augmented_images=100):
    """
    Perform a single training step on one image.
    """
    model, processor = get_model_and_processor()
    model.train()
    if FREEZE_QFORMER:
        for param in model.blip2.qformer.parameters():
            param.requires_grad = False
    if FREEZE_SHARED:
        for param in model.shared_features.parameters():
            param.requires_grad = False
    criterion = WeightedMultiHeadLoss(movement_weight=movement_weight, genre_weight=genre_weight, form_weight=form_weight).to(MAIN_DEVICE)


    augmented_pixel_values = augment_annotated_image(
        image, processor, max_zoom=16, show_debug=False, max_images=max_augmented_images)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_loss = 0.0
    for zoom_level, images in augmented_pixel_values.items():
        number_of_images = len(images)
        criterion.set_zoom_level(zoom_level)
        for i in range(0, number_of_images, batch_size):
            batch_images = images[i : i + batch_size]
            pixel_values = processor(images=batch_images, return_tensors="pt").pixel_values.to(MAIN_DEVICE)
            target_tensor = torch.tensor([target]*len(batch_images), dtype=torch.float32).to(MAIN_DEVICE)
            outputs = model(pixel_values)
            loss, loss_dict = criterion(outputs, target_tensor)
            print(f"Backprop on {image_id} with {zoom_level:.4f} zoom: Loss = Movement {loss_dict['movement']:.4f}, Genre {loss_dict['genre']:.4f}, Form {loss_dict['form']:.4f}, batch index {i}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    average_loss = total_loss / len(augmented_pixel_values)
    torch.cuda.empty_cache()
    print(f"Backward pass completed on image {image_id} | Average Loss: {average_loss:.4f}")
    return average_loss


# %%
# annotated training using expert-labeled data
def annotated_training(num_epochs=5, max_augmented_images=100,
                       movement_weight=0.05, genre_weight=0.05, form_weight=1.1):

    from annotater.backend.model_services import load_PIL_image, get_labels_created_dict
    print("="*60 + "ANNOTATED TRAINING" + "="*60)
    labels_created: dict[str, list[float]] = get_labels_created_dict()
    
    model = BLIP2MultiHeadRegression(
        blip2,
        use_form_head=True,
        train_qformer=True,
        train_vision=False
    )
    load_model_from_latest(model)

    _, test_loader = create_train_test_loaders(
        batch_size_train=32,
        batch_size_test=32,
        test_percentage=0.1,
    )
    test_criterion = WeightedMultiHeadLoss(
        use_form=False
    ).to(MAIN_DEVICE)

    total_loss = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
    
        epoch_loss = 0.0
        completed_images = 0
        for image_id, target in labels_created.items():
            try:
                image = load_PIL_image(image_id)
            except Exception as e:
                print(f"Skipping image {image_id} due to error: {e}")
                continue
            epoch_loss += backward_single_image(
                image,
                target,
                lr=1e-5,
                batch_size=4,
                image_id=image_id,
                max_augmented_images=max_augmented_images,
                movement_weight=movement_weight,
                genre_weight=genre_weight,
                form_weight=form_weight,
            )
            completed_images += 1
            print(f"{completed_images + 1}/{len(labels_created)} images processed in epoch {epoch}")

        avg_epoch_loss = epoch_loss / len(labels_created)
        total_loss += avg_epoch_loss
        # --- Training ---
        print(f"Epoch {epoch} training complete | Avg Loss: {avg_epoch_loss:.4f}")

        print("validating on test set to ensure not forgetting preliminary training")
        val_loss = test_epoch(
            model,
            test_loader,
            test_criterion,
            MAIN_DEVICE,
            processor
        )
        file_name = f"model_annotated_epoch_{epoch}_valLoss{val_loss:.4f}"
        save_progress(model, file_name)

    print(f"training complete | Avg Loss over {num_epochs} epochs: {total_loss / num_epochs:.4f}")
    print("\n" + "="*30, "ANNOTATED TRAINING COMPLETE", "="*30)
        # we need to make sure that the model is not forgetting the preliminary training
if ANNOTATED_TRAINING:
    annotated_training()
