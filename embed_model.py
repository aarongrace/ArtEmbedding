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
try:
    with open(".env", "r") as f:
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
from pathlib import Path

if PLATFORM == "PC":
    VISION_DEVICE = "cpu" # not enough GPU memory on laptop for full vision model
    MAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRELIM_TRAINING = False # not prelim training on laptop

elif PLATFORM == "IDAS":
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available on IDAS. Please restart with GPU access.")
    VISION_DEVICE = "cuda"
    MAIN_DEVICE = "cuda"
    PRELIM_TRAINING = True
else:
    raise ValueError(f"Unknown PLATFORM: {PLATFORM}")

print(f"Vision device: {VISION_DEVICE}, Main device: {MAIN_DEVICE}, Prelim training: {PRELIM_TRAINING}")
BASE_DIR = Path.cwd()
IMGS_DIR = BASE_DIR / "paintings"
PRELIM_METADATA_DIR = BASE_DIR / "metadata" / "paintings_metadata_with_rough_groundtruth.json"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
print(f"Base directory: {BASE_DIR}")

# vector dimensions
MOVEMENT_DIM = 6
GENRE_DIM = 6
STYLE_DIM = 6

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
    model.style_head.load_state_dict(state_dict["style_head"])
    
    if "qformer" in state_dict:
        model.blip2.qformer.load_state_dict(state_dict["qformer"])
        print(" Loaded Q-Former weights")
    
    print(f" Loaded weights from {latest_check_point}")


def save_progress(model, file_name):
    checkpoint_file = os.path.join(CHECKPOINTS_DIR, f"{file_name}.pt")

    state_dict = {
        "shared_features": model.shared_features.state_dict(),
        "movement_head": model.movement_head.state_dict(),
        "genre_head": model.genre_head.state_dict(),
        "style_head": model.style_head.state_dict(),
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
def get_split( total_length: int, test_percentage: float = 0.2, 
              split_file: str = BASE_DIR/"data_split.json", seed: int = 42,):
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
            all_splits = json.load(f)
    else:
        all_splits = {}

    key = str(total_length)

    if key in all_splits:
        split_data = all_splits[key]
        print(f"Loaded existing split for length {total_length}: "
              f"{len(split_data['train'])} train, {len(split_data['test'])} test")
        return split_data["train"], split_data["test"]

    # Otherwise, generate a new split
    random.seed(seed)
    indices = list(range(total_length))
    random.shuffle(indices)
    num_test = int(total_length * test_percentage)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    all_splits[key] = {
        "train": train_indices,
        "test": test_indices
    }
    with open(split_file, "w") as f:
        json.dump(all_splits, f, indent=2)

    print(f"Created new split for length {total_length}: "
          f"{len(train_indices)} train, {len(test_indices)} test → saved to {split_file}")
    return train_indices, test_indices



# %%
# dataloader creation for preliminary training; lazy loading is neeeded due to dataset size
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


def create_train_test_loaders( imgs_directory_path, pretraining_metadata_path, 
                              batch_size_train=32, batch_size_test=32, test_percentage=0.1):
    print("="*80, "LOADING DATASET", "="*80)
    image_paths = []
    image_ids = []
    all_files = sorted(os.listdir(imgs_directory_path))
    
    for file_name in all_files:
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(imgs_directory_path, file_name)
            image_paths.append(path)
            # Extract ID from filename (first part before underscore)
            image_ids.append(file_name.split("_")[0])
    print(f"Found {len(image_paths)} image files in {imgs_directory_path}")
    
    with open(pretraining_metadata_path, 'r', encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} paintings")
    
    targets = []
    valid_paths = []
    matched_count = 0
    for path, img_id in zip(image_paths, image_ids):
        if img_id in metadata and "rough_groundtruth" in metadata[img_id]:
            target = metadata[img_id]["rough_groundtruth"]
            targets.append(target)
            valid_paths.append(path)
            matched_count += 1
    if matched_count == 0:
        raise ValueError("No images matched with metadata! Check your image IDs and metadata format.")
    else:
        print(f"Matched {matched_count}/{len(image_paths)} images with valid targets")

    # ensuring all targets have the same dimension
    first_target = targets[0]
    target_dim = len(first_target)
    for i, t in enumerate(targets[:3]):
        if len(t) != target_dim:
            raise ValueError(f"Inconsistent target dimensions: image {i} has {len(t)}, expected {target_dim}")
    
    # split into train and test sets
    num_images = len(valid_paths)
    train_indices, test_indices = get_split(total_length=num_images, test_percentage=test_percentage)

    train_paths = [valid_paths[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]
    test_paths = [valid_paths[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]
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
        imgs_directory_path=IMGS_DIR,
        pretraining_metadata_path=PRELIM_METADATA_DIR,
        batch_size_train=32,
        batch_size_test=32,
        test_percentage=0.1,
    )


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
                 use_style_head=True,
                 train_qformer=False,
                 train_vision=False):
        super().__init__()

        # --- Core model ---
        self.blip2 = blip2_model
        self.use_style_head = use_style_head

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
        print(f"Use style head: {use_style_head}")

        # --- Shared feature extraction ---
        self.shared_features = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
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

        # --- Style head (always defined, but only used if enabled) ---
        self.style_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, STYLE_DIM)
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
        style_scores = self.style_head(shared_features)

        outputs = {
            'movement': movement_scores,
            'genre': genre_scores,
            'style': style_scores,
        }
        return outputs



# %%
# Multi-head weighted loss function
class WeightedMultiHeadLoss(nn.Module):
    def __init__(self, movement_weight=1.0, genre_weight=1.0, style_weight=1.0, use_style=True):
        super().__init__()
        self.movement_weight = movement_weight
        self.genre_weight = genre_weight
        self.style_weight = style_weight
        self.use_style = use_style

        # MSE loss for continuous targets
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, predictions, targets, confidences=None):
        """
        Args:
            predictions: dict with 'movement', 'genre', 'style' (raw outputs)
            targets: tensor [batch, total_dim] (target values)
            confidences: dict with confidence scores (optional)
        """
        # Split targets by head dimensions
        movement_target = targets[:, :MOVEMENT_DIM]
        genre_target    = targets[:, MOVEMENT_DIM : MOVEMENT_DIM + GENRE_DIM]

        # --- Movement loss ---
        movement_loss = self.mse(predictions['movement'], movement_target)
        if confidences is not None and 'movement' in confidences:
            movement_loss = movement_loss * confidences['movement']
        movement_loss = movement_loss.mean() * self.movement_weight

        # --- Genre loss ---
        genre_loss = self.mse(predictions['genre'], genre_target)
        if confidences is not None and 'genre' in confidences:
            genre_loss = genre_loss * confidences['genre']
        genre_loss = genre_loss.mean() * self.genre_weight

        # --- Total loss ---
        total_loss = movement_loss + genre_loss
        loss_dict = {'movement': movement_loss.item(), 'genre': genre_loss.item()}

        # --- Style loss (optional) ---
        if self.use_style:
            style_target = targets[:, MOVEMENT_DIM + GENRE_DIM :]
            style_loss = self.mse(predictions['style'], style_target)
            if confidences is not None and 'style' in confidences:
                style_loss = style_loss * confidences['style']
            style_loss = style_loss.mean() * self.style_weight
            total_loss += style_loss
            loss_dict['style'] = style_loss.item()

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
    num_epochs=5,
    early_stopping_patience=2
):
    print("="*60 + "PRELIMINARY TRAINING" + "="*60)

    # ---------------- Model setup ----------------
    model = BLIP2MultiHeadRegression(
        blip2,
        use_style_head=False,
        train_qformer=True,
        train_vision=False
    )
    load_model_from_latest(model)

    criterion = WeightedMultiHeadLoss(
        movement_weight=1.0,
        genre_weight=1.0,
        use_style=False
    ).to(MAIN_DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
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
        num_epochs=5,
        early_stopping_patience=5
    )

# %%
# web access functions
import torch
from transformers import Blip2Processor
from augmentation import augment_annotated_images

# --- Global variables for lazy loading ---
_model, _processor = None, None

def initialize_model_for_webaccess():
    """
    Initialize the BLIP2 multi-head regression model and processor.
    Loads the latest checkpoint if available.
    """
    model = BLIP2MultiHeadRegression(
        blip2,
        use_style_head=True,
        train_qformer=True,
        train_vision=False
    )

    load_model_from_latest(model)
    model.eval()

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

    print(f"Outputs: movement -> {outputs['movement'].shape}, genre -> {outputs['genre'].shape}, style -> {outputs['style'].shape}")
    # Move each head to CPU and convert to list
    embeddings = torch.cat([ outputs['movement'], outputs['genre'], outputs['style'] ], 
                           dim=1).cpu().tolist()
    

    print(f"Forward pass completed on {len(images)} images")
    return embeddings

def backward_single_image(image, target, lr=1e-5):
    """
    Perform a single training step on one image.
    """
    model, processor = get_model_and_processor()
    criterion = WeightedMultiHeadLoss(movement_weight=1.0, genre_weight=1.0, use_style=True).to(MAIN_DEVICE)

    augmented_images, augmented_targets = augment_annotated_images([image], [target])
    print(f"Augmented to {len(augmented_images)} images for training")
    

    model.train()
    inputs = processor(images=augmented_images, return_tensors="pt").pixel_values.to(MAIN_DEVICE)
    target_tensor = torch.tensor(augmented_targets, dtype=torch.float32).to(MAIN_DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    outputs = model(inputs)
    print("Model outputs obtained", outputs.keys())
    loss, loss_dict = criterion(outputs, target_tensor)

    print("Backward pass with loss:", loss.item(), loss_dict)

    loss.backward()
    optimizer.step()


    return loss.item(), loss_dict

