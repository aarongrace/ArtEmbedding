import os
import json
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
            

def augment_batch_paths(image_paths, targets, processor):
    """
    Load images, create flipped versions, and return pixel values.
    
    Args:
        image_paths: List of file paths to images
        targets: List of target vectors
        processor: BLIP2 processor
        
    Returns:
        pixel_values: Tensor of shape [batch_size*2, 3, H, W]
        doubled_targets: List with doubled targets
    """
    images = []
    doubled_targets = []
    
    for img_path, target in zip(image_paths, targets):
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Add original
        images.append(img)
        doubled_targets.append(target)
        
        # Add flipped
        flipped_img = ImageOps.mirror(img)
        images.append(flipped_img)
        doubled_targets.append(target)
    
    # Process all images at once
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    
    return pixel_values, doubled_targets

def augment_annotated_images(image_array, targets):
    """
    Augments images and their corresponding target vectors by adding horizontally flipped versions.
    
    Args:
        image_array: List of PIL Image objects
        targets: Numpy array or list of target vectors (one per image)
    
    Returns:
        augmented_images: List of PIL Images (original + flipped for each)
        augmented_targets: Numpy array of target vectors (duplicated for original + flipped)
    """
    augmented_images = []
    augmented_targets = []
    
    for img, target in zip(image_array, targets):
        # Add original image and target
        augmented_images.append(img)
        augmented_targets.append(target)

        # Create and add flipped image and target
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(flipped_img)
        augmented_targets.append(target)
    
    # Convert targets to numpy array
    augmented_targets = np.array(augmented_targets, dtype=np.float32)

    print(f"Augmentation complete:")
    print(f"  Original images: {len(image_array)}")
    print(f"  Augmented images: {len(augmented_images)}")
    print(f"  Target shape: {augmented_targets.shape}")

    return augmented_images, augmented_targets




def test_augmentation(augmented_imgs, augmented_ids, targets, pair_index=0):
    """
    Test and visualize the augmentation by showing an original-flipped pair.
    
    Args:
        augmented_imgs: List of augmented images
        augmented_ids: List of augmented image IDs
        targets: Numpy array of targets
        pair_index: Which pair to display (0 = first pair, 1 = second pair, etc.)
    """
    # Calculate indices for the pair
    original_idx = pair_index * 2
    flipped_idx = pair_index * 2 + 1
    
    # Display images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(augmented_imgs[original_idx])
    axes[0].set_title(f"Original - ID: {augmented_ids[original_idx]}")
    axes[0].axis('off')
    
    axes[1].imshow(augmented_imgs[flipped_idx])
    axes[1].set_title(f"Flipped - ID: {augmented_ids[flipped_idx]}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show targets
    print(f"Target for original: {targets[original_idx]}")
    print(f"Target for flipped: {targets[flipped_idx]}")
    print(f"Targets are identical: {np.array_equal(targets[original_idx], targets[flipped_idx])}")
    print(f"IDs match: {augmented_ids[original_idx] == augmented_ids[flipped_idx]}")
