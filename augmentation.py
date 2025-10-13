import os
import json
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
            
            
def augment_images_for_pretraining(image_array, image_ids, pretraining_metadata):
    """
    Augments images by adding horizontally flipped versions.
    
    Args:
        image_array: List of PIL Image objects
        image_ids: List of image IDs (e.g., ['000001', '000002', ...])
        pretraining_metadata: Dictionary with image metadata including pretraining_groundtruth
    
    Returns:
        augmented_images: List of PIL Images (original + flipped for each)
        augmented_image_ids: List of image IDs (duplicated for original + flipped)
        augmented_targets: Numpy array of target vectors (duplicated for original + flipped)
    """
    augmented_images = []
    augmented_image_ids = []
    augmented_targets = []
    
    for img, img_id in zip(image_array, image_ids):
        metadata = pretraining_metadata.get(img_id, None)
        if metadata is not None:
            target = metadata.get('pretraining_groundtruth', None)
            if target is not None:
                # Add original image, id, and target
                augmented_images.append(img)
                augmented_image_ids.append(img_id)
                augmented_targets.append(target)
                
                # Create flipped image
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Add flipped image, same id, and same target
                augmented_images.append(flipped_img)
                augmented_image_ids.append(img_id)
                augmented_targets.append(target)
            else:
                print(f"Warning: No pretraining_groundtruth for image {img_id}")
        else:
            print(f"Warning: No metadata found for image {img_id}")
    
    # Convert targets to numpy array
    augmented_targets = np.array(augmented_targets, dtype=np.float32)
    
    print(f"Augmentation complete:")
    print(f"  Original images: {len(image_array)}")
    print(f"  Augmented images: {len(augmented_images)}")
    print(f"  Augmented image_ids: {len(augmented_image_ids)}")
    print(f"  Target shape: {augmented_targets.shape}")
    
    return augmented_images, augmented_image_ids, augmented_targets


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
