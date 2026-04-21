#!/usr/bin/env python3
"""
One-time setup script: installs SAM 2 and downloads the checkpoint.
Run this before starting the app:  python setup_model.py
"""

import os
import subprocess
import sys
import urllib.request

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CHECKPOINT_NAME = "sam2.1_hiera_base_plus.pt"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
)


def install_sam2():
    """Install SAM 2 from GitHub source with CUDA build disabled."""
    try:
        import sam2  # noqa: F401

        print("[OK] SAM 2 is already installed.")
        return
    except ImportError:
        pass

    print("[...] Installing SAM 2 from GitHub (this may take a minute)...")
    env = os.environ.copy()
    env["SAM2_BUILD_CUDA"] = "0"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/facebookresearch/sam2.git",
        ],
        env=env,
    )
    print("[OK] SAM 2 installed successfully.")


def download_checkpoint():
    """Download the SAM 2.1 Hiera Base+ checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"[OK] Checkpoint already exists ({size_mb:.0f} MB): {checkpoint_path}")
        return

    print(f"[...] Downloading {CHECKPOINT_NAME}...")

    def progress_hook(block_num, block_size, total_size):
        """Report download progress to stdout during checkpoint fetch."""
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(
                f"\r     {mb_down:.1f} / {mb_total:.1f} MB ({pct:.0f}%)",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(
        CHECKPOINT_URL, checkpoint_path, reporthook=progress_hook
    )
    print()

    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"[OK] Checkpoint downloaded ({size_mb:.0f} MB): {checkpoint_path}")


if __name__ == "__main__":
    install_sam2()
    download_checkpoint()
    print("\nSetup complete! You can now run: python app.py")
