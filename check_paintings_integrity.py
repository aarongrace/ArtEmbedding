import json
import os
import re

# Load metadata
metadata_file = os.path.join('metadata', 'paintings_metadata.json')
with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"Total paintings in metadata: {len(metadata)}")

# Sanitize filename
def sanitize_filename(s):
    # Replace slashes with em dash
    s = s.replace("/", "—")
    # Remove literal %
    s = s.replace("%", "_")
    # Replace forbidden characters
    for c in '\\:*?"<>|':
        s = s.replace(c, "_")
    # Lowercase
    return s.lower()

# Build filename
def build_filename(id, painting):
    url_ending = painting.get("url", "unknown").split("/en/")[-1]
    return sanitize_filename(f"{id}_{url_ending}")

filename_to_id = {build_filename(id, painting) + ".jpg": id for id, painting in metadata.items()}

# Generate expected filenames
expected_files = {build_filename(id, painting) + ".jpg" for id, painting in metadata.items()}
print(f"Total expected files: {len(expected_files)}")
print(f"Sample expected filenames: {list(expected_files)[:5]}")

# Read local files
paintings_dir = 'paintings'
local_files = {f for f in os.listdir(paintings_dir) if os.path.isfile(os.path.join(paintings_dir, f))}
print(f"Total local files: {len(local_files)}")
print(f"Sample local filenames: {list(local_files)[:5]}")

# Metadata entries with no local file
entry_no_local = expected_files - local_files
print(f"Metadata entries with no local file: {len(entry_no_local)}")
print(entry_no_local)

# Local files with no metadata entry (optional)
local_no_entry = local_files - expected_files
print(f"Files in local directory with no metadata entry: {len(local_no_entry)}")
print(local_no_entry)

# delete the files 
for filename in local_no_entry:
    file_path = os.path.join(paintings_dir, filename)
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

# for filename in entry_no_local:
#     id_to_remove = filename_to_id.get(filename)
#     if id_to_remove and id_to_remove in metadata:
#         print(f"Removing metadata entry for ID {id_to_remove} (filename: {filename})")
#         del metadata[id_to_remove]

# with open(metadata_file, 'w', encoding='utf-8') as f:
#     json.dump(metadata, f, indent=4)