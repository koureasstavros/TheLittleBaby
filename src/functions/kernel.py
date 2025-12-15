#########################
# Reload Functions
# Author: Koureas Stavros
#########################

import os
import sys
import shutil
import importlib

def terminate():
    print("Terminating...")
    sys.exit(1)

def import_module(module_name):
    if module_name not in sys.modules:
        print(f"📦 Importing: {module_name}")
        importlib.import_module(module_name)
    return sys.modules[module_name]

def remove_module(module_name):
    if module_name in sys.modules:
        print(f"🗑️ Removing: {module_name}")
        del sys.modules[module_name]

def terminate_module(module_name):
    if module_name in sys.modules:
        print(f"🛑 Terminating: {module_name}")
        del sys.modules[module_name]

def reload_modules(base_dir):
    modules_to_reload = []
    base_dir = os.path.abspath(base_dir)

    # First pass: identify modules to reload and delete __pycache__
    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            if dir == '__pycache__':
                pycache_path = os.path.join(root, dir)
                shutil.rmtree(pycache_path)
                print(f"🗑️ Deleted: {pycache_path}")
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Build relative path from base_dir
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                module_path = rel_path.replace(os.sep, ".").replace(".py", "")
                full_module_name = f"src.{module_path}"

                if full_module_name in sys.modules:
                    modules_to_reload.append(full_module_name)

    # Second pass: remove modules
    for module_name in modules_to_reload:
        print(f"🗑️ Removing: {module_name}")
        del sys.modules[module_name]
    
    # Third pass: import modules (dependencies resolve correctly)
    for module_name in modules_to_reload:
        print(f"📦 Importing: {module_name}")
        importlib.import_module(module_name)