
import numpy as np
from model_services import add_to_seen_list

# Example: initialize images
images = [{"id": "img1", "url": "/images/img1.png"}]
paintings_folder_path = ""

# Forward pass using notebook code
def forward_pass(image_id: str):
    # Your notebook code here
    # For example, call a model or process the vector
    random_17d_vector = np.random.rand(17).tolist()
    add_to_seen_list(image_id)
    return random_17d_vector

# Backprop using notebook code
def backprop(image_id: str, vector: list):
    # Notebook backprop logic here
    print(f"Backprop for {image_id} with vector {vector}")

# Random image
def get_random_image():
    import random
    return random.choice(images)
