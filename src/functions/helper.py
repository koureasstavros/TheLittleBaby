#########################
# Helper Functions
# Author: Koureas Stavros
#########################

import os
import sys
import importlib
import subprocess

def load_package(packages_needed):
    installed_packages = {dist.metadata['Name'].lower() for dist in importlib.metadata.distributions()}
    for package in packages_needed:
        if package.lower() not in installed_packages:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def get_cpu_properties(mp):
    if sys.platform.startswith("win"):
        output = os.popen("wmic cpu get name /format:list").read()
        for line in output.strip().splitlines():
            if "=" in line:
                _, value = line.split("=", 1)
                name = value.strip()
                break
    elif sys.platform.startswith("linux"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    name = line.strip().split(":")[1].strip()
                    break
    elif sys.platform.startswith("darwin"):
        name = os.popen("sysctl -n machdep.cpu.brand_string").read().strip()

    # Return the name as a string
    return str(name)

def get_gpu_properties(mp, selected_core):
    if mp.cuda.runtime.getDeviceCount() > 0:
        return str(mp.cuda.runtime.getDeviceProperties(int(selected_core))["name"].decode("utf-8"))