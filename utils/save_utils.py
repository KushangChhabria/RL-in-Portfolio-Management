import json
import os
from datetime import datetime

def save_metadata(path, metadata: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_metadata(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
