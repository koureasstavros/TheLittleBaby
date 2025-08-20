#########################
# Helper Functions
# Author: Koureas Stavros
#########################

import sys
import importlib
import subprocess
from dotenv import load_dotenv

def load_environment():
    load_dotenv(override=True)

def load_package(packages_needed):
    installed_packages = {dist.metadata['Name'].lower() for dist in importlib.metadata.distributions()}
    for package in packages_needed:
        if package.lower() not in installed_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])