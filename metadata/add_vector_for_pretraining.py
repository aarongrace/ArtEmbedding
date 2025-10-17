import json

vector_properties = [
    # Movements (6)
    "movement_baroque",
    "movement_rococo",
    "movement_neoclassicism",
    "movement_romanticism",
    "movement_realism",
    "movement_impressionism",

    # Genres (6)
    "genre_historical",
    "genre_religious",
    "genre_mythological",
    "genre_everyday_life",
    "genre_landscape",
    "genre_portrait",

    # Style dimensions (6)
    "style_naturalistic",   # Stylized ↔ Naturalistic
    "style_dynamic",        # Still ↔ Dynamic
    "style_brushstrokes",   # Smooth ↔ Loose
    "style_complexity",     # Simple → Complex
    "style_balance",        # Lopsided → Balanced
    "style_emotionality"    # Unemotional → Emotional
]


def metadata_to_vector(metadata):
    """
    Convert WikiArt metadata to an 18-dim vector with predefined style axes.
    Returns None if genre or movement is missing or invalid.
    """

    # Allowed genres and movements
    allowed_genres = [
        "battle", "history", "genre", "religious painting",
        "landscape", "portrait", "mythological", "literary"
    ]
    allowed_movements = [
        "baroque", "rococo", "neoclassicism", "romanticism",
        "orientalism", "realism", "impressionism"
    ]

    # Check for missing metadata
    if "genres" not in metadata or "styles" not in metadata:
        return None

    # Initialize default vector
    new_vector = [
        0.0,  # Baroque
        0.0,  # Rococo
        0.0,  # Neoclassicism
        0.0,  # Romanticism
        0.0,  # Realism
        0.0,  # Impressionism

        0.0,  # Historical
        0.0,  # Religious
        0.0,  # Mythological
        0.0,  # Everyday life
        0.0,  # Landscape
        0.0,  # Portrait
    ]

    # Process movement
    movement_set = [m.lower() for m in metadata.get("styles", [])]
    movement_index_map = {
        "baroque": 0,
        "rococo": 1,
        "neoclassicism": 2,
        "romanticism": 3,
        "realism": 4,
        "impressionism": 5
    }
    movement_found = False
    for movement in movement_index_map.keys():
        if movement in movement_set:
            movement_found = True
            new_vector[movement_index_map[movement]] = 1.0
    if not movement_found:
        return None

    # Process genre
    genre_set = [g.lower() for g in metadata.get("genres", [])]
    genre_index_map = {
        "history": 6,
        "battle": 6,
        "religious painting": 7,
        "mythological": 8,
        "genre": 9,         # Everyday life
        "landscape": 10,
        "portrait": 11,
    }
    genre_found = False
    for genre in genre_index_map.keys():
        if any(genre in g for g in genre_set):
            genre_found = True
            new_vector[genre_index_map[genre]] = 1.0
    if not genre_found:
        return None

    return new_vector

from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent
print(f"Base directory for metadata: {BASE_DIR}")
# --- Execution ---
raw_metadata_path = os.path.join(BASE_DIR, 'paintings_metadata.json')
with open(raw_metadata_path, 'r', encoding="utf-8") as f:
    paintings_metadata = json.load(f)

print(f"Total paintings in raw metadata: {len(paintings_metadata)}")
print(f"Sample metadata entry:\n{json.dumps(paintings_metadata[paintings_metadata.keys().__iter__().__next__()], indent=4)}")

converted_metadata = {}
skipped_count = 0

for i in paintings_metadata:
    item = paintings_metadata[i]
    image_id = item["id"]
    vector = metadata_to_vector(item)

    if vector is not None:
        item["rough_groundtruth"] = vector
        item.pop("id", None)  # Remove the id field since it's now the key
        converted_metadata[image_id] = item
    else:
        skipped_count += 1
        if "literary painting" in item.get("genres"):
            print(f"[SKIPPED] {image_id} has genres {item.get('genres')} .")
        else :
            print(f"[SKIPPED] {image_id} -> missing/invalid genre or style.")

print(f"\nFinished processing.")
print(f"✅ Converted: {len(converted_metadata)}")
print(f"❌ Skipped: {skipped_count}")

output_path = os.path.join(BASE_DIR, 'paintings_metadata_with_rough_groundtruth.json')
with open(output_path, 'w', encoding="utf-8") as f:
    json.dump(converted_metadata, f, ensure_ascii=False, indent=4)

print(f"Saved processed metadata to {output_path}")
