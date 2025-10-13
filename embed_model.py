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
#     display_name: Python (my_venv)
#     language: python
#     name: my_venv
# ---

# %%
MOVEMENT_DIM = 5
GENRE_DIM = 5
STYLE_DIM = 6


# %%


# get the images
from PIL import Image
import os, json, torch

MAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISION_DEVICE = MAIN_DEVICE
try:  # Check if running in Colab
    from google.colab import drive
    IN_COLAB = True
    print("running in Google Colab")
    mount_path = '/content/drive'
    if not os.path.exists(mount_path):
        drive.mount(mount_path)
    imgs_directory_path = '/content/drive/MyDrive/ArtEmbed'
    pretraining_metadata = '/content/drive/MyDrive/ArtEmbed/wikiart_metadata_with_pretraining_groundtruth.json'

except ImportError:  # Not Colab
    from pathlib import Path
    IN_COLAB = False

    try:
        BASE_DIR = Path(__file__).resolve().parent  # works in scripts
        print("running from laptop, probably")
        VISION_DEVICE = "cpu" # not even GPU mem on laptop
    except NameError:
        BASE_DIR = Path.cwd()  # fallback for notebooks
        print("running from IDAS, probably")

    imgs_directory_path = BASE_DIR / "paintings"
    pretraining_metadata = BASE_DIR / "metadata" / "wikiart_metadata_with_pretraining_groundtruth.json"


def load_image_from_drive():
  image_array = []
  image_names = []
  image_ids =[]

  all_files = sorted(os.listdir(imgs_directory_path))
  for file_name in all_files:
      if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
          path = os.path.join(imgs_directory_path, file_name)
          img = Image.open(path).convert("RGB")
          image_array.append(img)
          image_names.append(file_name)
          image_ids.append(file_name.split("_")[0])

  print(f"Found {len(image_array)} images. Image ids: {image_ids}")
  return image_array, image_ids

def load_pretraining_metadata():
    with open(pretraining_metadata, 'r', encoding="utf-8") as f:
        metadata = json.load(f)
    # print(metadata.keys())
    print(f"Found metadata for {len(metadata)} paintings.")
    return metadata



# %%

# --- Import libraries ---
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# --- Load BLIP-2 model and processor ---
model_name = "Salesforce/blip2-flan-t5-xl"
local_model_path =  BASE_DIR / "blip2_model"

if os.path.exists(local_model_path):
    print("Loading model from local directory...")
    processor = Blip2Processor.from_pretrained(local_model_path, use_fast=True)
    print("Processor loaded")
    blip2 = Blip2ForConditionalGeneration.from_pretrained(local_model_path)
    print("Model loaded")
else:
    print("Downloading model from Hugging Face...")
    processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
    blip2 = Blip2ForConditionalGeneration.from_pretrained(model_name)

    # Save to local directory for future use
    processor.save_pretrained(local_model_path)
    blip2.save_pretrained(local_model_path)

blip2.to("cpu")  # Load model on CPU to avoid GPU memory issues
print(f"Loaded model on cpu")

# Freeze vision encoder to save memory; we are not training the vision encoder
for param in blip2.vision_model.parameters():
    param.requires_grad = False


# %%


from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(image_list, target_list, processor, device, batch_size=4, shuffle=True):
    # Convert images to pixel values tensors
    pixel_values_tensor = torch.stack([
        processor(images=img, return_tensors="pt").pixel_values.squeeze(0) 
        for img in image_list
    ])  # [N, 3, H, W]

    # Convert targets to tensor
    targets_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in target_list])  # [N, total_dims]

    # Create TensorDataset
    dataset = TensorDataset(pixel_values_tensor, targets_tensor)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

    # Wrap batches with device transfer
    def device_loader():
        for batch_pixel_values, batch_targets in dataloader:
            yield batch_pixel_values.to(device, non_blocking=True), batch_targets.to(device, non_blocking=True)

    print(f"Created DataLoader with {len(dataloader)} batches of size {batch_size}")

    return device_loader()



# %%


def print_gpu_mem(prefix="GPU"):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2   # MB
        reserved = torch.cuda.memory_reserved() / 1024**2     # MB
        print(f"{prefix} Memory — Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    else:
        print("CUDA not available")


# %%


from torch import nn

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

    def forward(self, images, return_features=False):
        """
        Forward pass with optional CPU/GPU split for vision model.

        Args:
            images: [batch_size, 3, H, W]
            return_features: If True, also return shared features

        Returns:
            dict with keys: 'movement', 'genre', 'style', 'combined', optionally 'features'
        """

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
        movement_scores = torch.sigmoid(self.movement_head(shared_features))
        genre_scores = torch.sigmoid(self.genre_head(shared_features))
        style_scores = torch.sigmoid(self.style_head(shared_features))

        outputs = {
            'movement': movement_scores,
            'genre': genre_scores,
            'style': style_scores,
            'combined': torch.cat([movement_scores, genre_scores, style_scores], dim=1)
        }

        if return_features:
            outputs['features'] = shared_features

        return outputs


class WeightedMultiHeadLoss(nn.Module):
    def __init__(self, movement_weight=1.0, genre_weight=0.7, style_weight=0.8, use_style=True):
        super().__init__()
        self.movement_weight = movement_weight
        self.genre_weight = genre_weight
        self.style_weight = style_weight
        self.use_style = use_style

    def forward(self, predictions, targets, confidences=None):
        """
        Args:
            predictions: dict with 'movement', 'genre', 'style'
            targets: tensor [batch, total_dim] (already prepared)
            confidences: dict with confidence scores (optional)
        """
        # Split targets using global dims
        movement_target = targets[:, :MOVEMENT_DIM]
        genre_target    = targets[:, MOVEMENT_DIM : MOVEMENT_DIM + GENRE_DIM]
        style_target    = targets[:, MOVEMENT_DIM + GENRE_DIM :]

        mse = nn.MSELoss(reduction='none')

        # Movement loss
        movement_loss = mse(predictions['movement'], movement_target)
        if confidences is not None and 'movement' in confidences:
            movement_loss = movement_loss * confidences['movement']
        movement_loss = movement_loss.mean() * self.movement_weight

        # Genre loss
        genre_loss = mse(predictions['genre'], genre_target)
        if confidences is not None and 'genre' in confidences:
            genre_loss = genre_loss * confidences['genre']
        genre_loss = genre_loss.mean() * self.genre_weight

        total_loss = movement_loss + genre_loss
        loss_dict = {'movement': movement_loss.item(), 'genre': genre_loss.item()}

        # Style loss
        if self.use_style:
            style_loss = mse(predictions['style'], style_target)
            if confidences is not None and 'style' in confidences:
                style_loss = style_loss * confidences['style']
            style_loss = style_loss.mean() * self.style_weight
            total_loss += style_loss
            loss_dict['style'] = style_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# %%


import time

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_images = 0

    start_time = time.time()

    for step, (pixel_values, targets) in enumerate(dataloader):
        batch_size = pixel_values.size(0)
        num_batches += 1
        num_images += batch_size

        pixel_values = pixel_values.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)  # [batch, total_dim]


        optimizer.zero_grad()
        predictions = model(pixel_values)
        loss, loss_dict = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        time_elapsed = time.time() - start_time
        # if step % 2 == 0:
        if True:
            print(f"Step {step} image number {num_images} time_elapsed {time_elapsed:.2f}s | Loss: {loss.item():.4f}")
            print_gpu_mem()

    end_time = time.time()
    epoch_time = end_time - start_time
    avg_loss = total_loss / num_batches
    print(f"Epoch complete | Avg Loss: {avg_loss:.4f}")
    print(f"Time: {epoch_time:.2f}s | Per batch: {epoch_time/num_batches:.2f}s | Per image: {epoch_time/num_images:.4f}s")

    return avg_loss



def test_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for pixel_values, targets in dataloader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)  # [batch, total_dim]

            predictions = model(pixel_values)
            loss, _ = criterion(predictions, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation complete | Avg Loss: {avg_loss:.4f}")
    return avg_loss


# %%


def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, save_path=None):
    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")

        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history["train_loss"].append(train_loss)

        # Validation
        if val_loader is not None:
            val_loss = test_epoch(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)

        # Save checkpoint
        if save_path is not None:
            checkpoint_file = f"{save_path}/model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Saved checkpoint: {checkpoint_file}")

    print("\nTraining complete")
    return history


# %%


import random
def split_train_test(image_list, targets, test_percentage=0.1):
    test_percentage = 0.1
    num_images = len(image_list)
    num_test = int(num_images * test_percentage)

    # Randomly sample indices for test set
    test_indices = random.sample(range(num_images), num_test)

    # Create test images and targets
    test_images = [image_list[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]

    # Optionally, remove test items from the training set
    train_images = [img for idx, img in enumerate(image_list) if idx not in test_indices]
    train_targets = [tgt for idx, tgt in enumerate(targets) if idx not in test_indices]

    train_loader = create_dataloader(
        train_images, train_targets, processor, MAIN_DEVICE, batch_size=16, shuffle=True
    )
    test_loader = create_dataloader(
        test_images, test_targets, processor, MAIN_DEVICE, batch_size=32, shuffle=False
    )
    return train_loader, test_loader


# %%


def pretrain_model():
    # PRETRAINING: No style head
    print("="*50)
    print("PRETRAINING MODE (no style head)")
    print("="*50)
    image_list, image_ids = load_image_from_drive()
    pretraining_metadata = load_pretraining_metadata()
    from augmentation import augment_images_for_pretraining
    image_list, image_ids, targets = augment_images_for_pretraining(image_list, image_ids, pretraining_metadata)   


    pretrain_model = BLIP2MultiHeadRegression( blip2,
        use_style_head=False, train_qformer=False, train_vision=False
    )
    pretrain_criterion = WeightedMultiHeadLoss( movement_weight=1.0, genre_weight=0.7,
        use_style=False,).to(MAIN_DEVICE)
    optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=1e-4)


    train_loader, test_loader = split_train_test(image_list, targets, test_percentage=0.1)
    save_dir = "./checkpoints"
    history = train_model(pretrain_model, train_loader, test_loader, optimizer, pretrain_criterion,
        MAIN_DEVICE, num_epochs=1, save_path=save_dir
    )
# pretrain_model()


# %%


import os
import glob
import torch
from transformers import Blip2Processor
from augmentation import augment_annotated_images

# --- Global variables for lazy loading ---
_model, _processor = None, None

def get_latest_checkpoint(checkpoint_dir="./checkpoints"):
    """Return the latest checkpoint path or None if none exist."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pt"))
    if not checkpoint_files:
        return None
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
    return checkpoint_files[-1]

import os
from datetime import datetime
import torch

BASE_DIR = "/path/to/your/project"  # replace with your BASE_DIR

def save_model_checkpoint(model):
    checkpoint_dir = os.path.join(BASE_DIR, ".checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Generate abbreviated timestamp (YYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # Build checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{timestamp}.pt")

    # Save model state
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


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

    latest_ckpt = get_latest_checkpoint()
    if latest_ckpt is not None:
        model.load_state_dict(torch.load(latest_ckpt, map_location="cpu"))
        print(f"Loaded model weights from {latest_ckpt}")
    else:
        print("No checkpoint found, using untrained weights.")

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

    embeddings = outputs["combined"].cpu().tolist()
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

