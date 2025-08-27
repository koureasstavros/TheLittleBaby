#########################
# Runtime Function
# Author: Koureas Stavros
#########################

import re
import os
import sys
import uuid
import json
from pathlib import Path as ph

debug = False

def is_valid_guid(guid_str):
    try:
        val = uuid.UUID(guid_str)
        return str(val) == guid_str  # Ensures exact format match
    except ValueError:
        return False
    
def is_debug():
    return debug

def pt_debug(text):
    if debug:
        print(f"[DEBUG] {text}")

def from_file(file_path, file_mode):
    try:
        if file_mode == "plain":
            with open(file_path, "r") as f:
                file = f.read()
                return file
        elif file_mode == "binary":
            with open(file_path, "rb") as f:
                file = f.read()
                return file
        elif file_mode == "json":
            with open(file_path, "r") as f:
                file = json.load(f)
                return file
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)

def towa_file(file_path, file_mode, content):
    try:
        if file_mode == "plain":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        elif file_mode == "binary":
            with open(file_path, "wb") as f:
                f.write(content)
        elif file_mode == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)

def remv_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)

def get_directory_files(folder_path, file_prefix):
    runtime_uuids = []
    # Define the directory
    directory = ph(f"{folder_path}")

    # Regex pattern for GUID
    pattern = re.compile(
        rf'^{file_prefix}_([a-fA-F0-9-]+(?:_finetuned)?)\.json$',
        re.IGNORECASE
    )

    # Find matching files and extract GUIDs
    for file in directory.glob(f'{file_prefix}_*.json'):
        filename = file.name  # Ensure we get the full filename
        match = pattern.match(filename)
        if match:
            runtime_uuids.append(match.group(1))

    print(f"Found runtime UUIDs: {runtime_uuids}")
    return runtime_uuids