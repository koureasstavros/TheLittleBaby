#########################
# Reload Functions
# Author: Koureas Stavros
#########################

import os
import sys
import importlib

def reload_modules(base_dir):
    base_dir = os.path.abspath(base_dir)
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Build relative path from base_dir
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                module_path = rel_path.replace(os.sep, ".").replace(".py", "")
                full_module_name = f"src.{module_path}"

                try:
                    if full_module_name in sys.modules:
                        print(f"🔁 Reloading: {full_module_name}")
                        importlib.reload(sys.modules[full_module_name])
                    else:
                        print(f"📦 Importing: {full_module_name}")
                        module = importlib.import_module(full_module_name)
                        importlib.reload(module)
                except Exception as e:
                    print(f"⚠️ Failed to reload {full_module_name}: {e}")