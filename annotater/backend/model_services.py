from pathlib import Path
import base64
import json
import random

# Get root folder of the project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # move up from backend/model.py to root
PAINTINGS_DIR = ROOT_DIR / "paintings"
METADATA_FILE = ROOT_DIR / "paintings_metadata.json"
USERS_STATE_FILE = ROOT_DIR / "users_state.json"
USER_NAME = "admin"  # Default user name for simplicity

GROUND_TRUTH_LABELS_FIELD_NAME = "ground_truths_created"

meta_data = None
def get_metadata() -> dict:
    global meta_data
    if meta_data is None:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return metadata

def get_metadata_by_id(image_id: str) -> dict:
    entry = get_metadata().get(image_id)
    if entry is None:
        raise ValueError(f"No metadata found for image ID: {image_id}")
    return get_metadata().get(image_id)


def get_user_state(user_id: str) -> dict:
    if not USERS_STATE_FILE.exists():
        return {}
    with open(USERS_STATE_FILE, "r", encoding="utf-8") as f:
        users_state = json.load(f)
    return users_state.get(user_id, {})

def get_seen_list(user_id: str = USER_NAME) -> list:
    user_state = get_user_state(user_id)
    return user_state.get("seen_paintings", [])

def add_to_seen_list(image_id: str, user_id: str = USER_NAME):
    if USERS_STATE_FILE.exists():
        with open(USERS_STATE_FILE, "r", encoding="utf-8") as f:
            users_state = json.load(f)
    else:
        raise FileNotFoundError("Users state file does not exist.")

    user_state = users_state.get(user_id, {})
    seen_list = user_state.get("seen_paintings", [])
    if image_id not in seen_list:
        seen_list.append(image_id)
        user_state["seen_paintings"] = seen_list
        users_state[user_id] = user_state

        with open(USERS_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(users_state, f, indent=4)

def add_to_labels_list(image_id: str, label_vector: list, user_id: str = USER_NAME):
    if USERS_STATE_FILE.exists():
        with open(USERS_STATE_FILE, "r", encoding="utf-8") as f:
            users_state = json.load(f)
    else:
        raise FileNotFoundError("Users state file does not exist.")

    user_state = users_state.get(user_id, {})
    labels_list = user_state.get(GROUND_TRUTH_LABELS_FIELD_NAME)
    if labels_list is None:
        raise ValueError(f"User state for {user_id} does not have field {GROUND_TRUTH_LABELS_FIELD_NAME}")
    if image_id in labels_list.keys():
        print(f"Warning: Overwriting existing label for image ID {image_id}")
    
    labels_list[image_id] = label_vector
    user_state[GROUND_TRUTH_LABELS_FIELD_NAME] = labels_list
    users_state[user_id] = user_state

    with open(USERS_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(users_state, f, indent=4)


def get_user_name() -> str:
    return USER_NAME

def set_user_name(name: str):
    global USER_NAME
    USER_NAME = name


all_image_paths = sorted(list(PAINTINGS_DIR.glob("*.jpg")))
def get_image_path(image_id: str, local: bool) -> str:
    image_id_num = int(image_id)
    print(f"Getting path for image ID: {image_id} -> {image_id_num}")
    if image_id_num < 1 or image_id_num > len(all_image_paths):
        raise ValueError(f"Invalid image ID: {image_id}")
    
    local_path = str(all_image_paths[image_id_num - 1])
    if local:
        return local_path
    else:
        file_name = local_path.split("\\")[-1]  # Adjust for Windows path separator
        return f"/paintings/{file_name}"


def get_64_encoded_image(image_id: str) -> str:
    image_path = get_image_path(image_id, local=True)
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string

def get_random_image_id():
    ids_in_folder = [p.stem.split("_")[0] for p in all_image_paths]
    seen_list = get_seen_list()
    possible_ids = [list(set(ids_in_folder) - set(seen_list))][0]
    # print(f"Possible IDs after filtering seen: {possible_ids}")
    random_index =  random.randint(0, len(possible_ids) - 1)
    return possible_ids[random_index]

