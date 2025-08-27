#########################
# Reload Functions
# Author: Koureas Stavros
#########################

import os
import sys
import shutil
import importlib

def terminate():
    print(f"Terminating...")
    sys.exit(1)

def import_module(module_name):
    if module_name not in sys.modules:
        print(f"📦 Importing: {module_name}")
        module = importlib.import_module(module_name)
        importlib.reload(module)
    return sys.modules[module_name]

def terminate_module(module_name):
    if module_name in sys.modules:
        print(f"🛑 Terminating: {module_name}")
        del sys.modules[module_name]

def reload_modules(base_dir):
    base_dir = os.path.abspath(base_dir)
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

                try:
                    if full_module_name in sys.modules:
                        print(f"🔁 Reloading: {full_module_name}")
                        del sys.modules[full_module_name]
                        module = importlib.import_module(full_module_name)
                        importlib.reload(sys.modules[full_module_name])
                    else:
                        print(f"📦 Importing: {full_module_name}")
                        module = importlib.import_module(full_module_name)
                        importlib.reload(module)
                except Exception as e:
                    print(f"⚠️ Failed to reload {full_module_name}: {e}")