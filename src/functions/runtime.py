#########################
# Runtime Function
# Author: Koureas Stavros
#########################

import uuid
import json

debug = False

def is_valid_guid(guid_str):
    try:
        val = uuid.UUID(guid_str)
        return str(val) == guid_str  # Ensures exact format match
    except ValueError:
        return False
    
def is_debug(text):
    if debug:
        print(f"[DEBUG] {text}")

def from_file(file_path, file_mode):
    try:
        if file_mode == "plain":
            with open(file_path, "r") as f:
                file = f.read()
                return file
        elif file_mode == "json":
            with open(file_path, "r") as f:
                file = json.load(f)
                return file
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        exit(1)
        

def to_file(file_path, file_mode, content):
    try:
        if file_mode == "plain":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        elif file_mode == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=4)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        exit(1)