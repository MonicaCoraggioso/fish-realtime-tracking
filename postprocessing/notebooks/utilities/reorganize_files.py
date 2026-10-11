"""
    Online tracking: position of the fish + tail points
    
    Author: Monica Coraggioso
"""

import shutil
import os
import time


def copy_file(src, dst, retries=5, delay=1):
    for i in range(retries):
        try:
            shutil.copy(src, dst)
            print(f"✅ File copied from {src} to {dst}")
            return
        except PermissionError as e:
            print(f"⚠️ Attempt {i + 1}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise PermissionError(f"❌ Failed to copy file from {src} to {dst} after {retries} attempts.")
