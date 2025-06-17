#!/usr/bin/env python3
"""
Download script for LongMemEval dataset

This script downloads the LongMemEval dataset from the official Google Drive source
and extracts it to the correct location for the evaluation script.
"""

import os
import sys
import json
from pathlib import Path
import tarfile

def install_gdown():
    """Install gdown if not available."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("Installing gdown...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        return gdown

def main():
    """Download LongMemEval dataset."""
    # Get the data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("LongMemEval Dataset Downloader")
    print("=" * 40)
    
    # Install gdown if needed
    try:
        gdown = install_gdown()
    except Exception as e:
        print(f"❌ Failed to install gdown: {e}")
        print("Please install manually: pip install gdown")
        return
    
    # Official Google Drive download link
    file_id = '1zJgtYRFhOh5zDQzzatiddfjYhFSnyQ80'
    url = f'https://drive.google.com/uc?id={file_id}'
    file_path = data_dir / 'longmemeval_data.tar.gz'
    
    print("📥 DOWNLOADING DATASET:")
    print(f"Source: Official Google Drive")
    print(f"URL: {url}")
    print(f"Destination: {file_path}")
    print()
    
    # Download the compressed dataset
    if not file_path.exists():
        try:
            print("Downloading longmemeval_data.tar.gz...")
            gdown.download(url, str(file_path), quiet=False)
            print("✅ Download completed!")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("You can try downloading manually from:")
            print(f"https://drive.google.com/file/d/{file_id}/view")
            return
    else:
        print(f"✅ '{file_path.name}' already exists, skipping download.")
    
    print()
    print("📦 EXTRACTING DATASET:")
    
    # Check if files already exist
    expected_files = [
        'longmemeval_oracle.json',
        'longmemeval_s.json', 
        'longmemeval_m.json'
    ]
    
    files_exist = all((data_dir / filename).exists() for filename in expected_files)
    
    if not files_exist:
        try:
            print("Extracting tar.gz file...")
            with tarfile.open(file_path, 'r:gz') as tar:
                # Extract to data directory
                tar.extractall(path=data_dir)
            print("✅ Extraction completed!")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return
    else:
        print("✅ Dataset files already exist, skipping extraction.")
    
    print()
    print("📋 VERIFYING FILES:")
    
    all_found = True
    total_size = 0
    
    for filename in expected_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"✅ {filename} - Found ({size_mb:.1f} MB)")
        else:
            print(f"❌ {filename} - Not found")
            all_found = False
    
    print()
    if all_found:
        print(f"🎉 SUCCESS! All dataset files downloaded and extracted ({total_size:.1f} MB total)")
        print()
        print("📊 DATASET VARIANTS:")
        print("• longmemeval_oracle.json - Oracle retrieval (easiest, for testing)")
        print("• longmemeval_s.json - Short version (~115k tokens, ~40 sessions)")  
        print("• longmemeval_m.json - Medium version (~500 sessions)")
        print()
        print("🚀 READY TO RUN EVALUATION:")
        print("cd eval/longmemeval")
        print("python evaluate_memorizz.py --dataset_variant oracle")
        print("python evaluate_memorizz.py --dataset_variant s")
        print("python evaluate_memorizz.py --dataset_variant m")
    else:
        print("⚠️  Some dataset files are missing after extraction.")
        print("Please check the extracted files or try downloading again.")
    
    # Clean up compressed file (optional)
    if file_path.exists() and all_found:
        try:
            file_path.unlink()
            print(f"🗑️  Cleaned up compressed file: {file_path.name}")
        except:
            pass  # Don't fail if cleanup doesn't work
    
    print(f"\n📂 Data directory: {data_dir}")
    print("📄 Dataset paper: https://arxiv.org/abs/2410.10813")

if __name__ == "__main__":
    main() 