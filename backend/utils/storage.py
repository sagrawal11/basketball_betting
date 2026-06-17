"""
Cloudflare R2 (S3-compatible) utilities for syncing ML artifacts.

IMPORTANT: Only syncs what the API actually needs at runtime (~173 MB),
NOT the full 11 GB training dataset. Training stays local.
"""
import os
import boto3
from datetime import datetime, timedelta
from pathlib import Path
from botocore.exceptions import ClientError
from botocore.client import Config
import pandas as pd


def get_r2_client():
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    # When set, point boto3 at a local S3-compatible store (MinIO) instead of R2.
    # This is a real S3 endpoint, not a mock; production R2 behavior is unchanged
    # whenever R2_ENDPOINT_URL is absent.
    endpoint = os.environ.get("R2_ENDPOINT_URL")

    if not all([access_key, secret_key]):
        return None

    if endpoint:
        # Local MinIO stand-in: needs explicit region + path-style addressing.
        return boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}),
        )

    # Production Cloudflare R2 (behavior identical to before this seam was added).
    if not account_id:
        return None
    return boto3.client(
        's3',
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )


def _upload_file(s3, local_path: Path, bucket: str, key: str) -> bool:
    """Upload a single file to R2."""
    try:
        s3.upload_file(str(local_path), bucket, key)
        return True
    except ClientError as e:
        print(f"  Upload failed for {key}: {e}")
        return False


def _list_remote_sizes(s3, bucket: str) -> dict:
    """Map remote key -> size (bytes) for every object in the bucket, for delta uploads."""
    sizes: dict = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def _get_active_player_slugs(player_data_dir: Path, max_days: int = 365) -> list[str]:
    """Return slugs of players who have played within max_days."""
    cutoff = datetime.now() - timedelta(days=max_days)
    active = []
    for d in player_data_dir.iterdir():
        if not d.is_dir():
            continue
        csv_path = d / f"{d.name}_data.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, usecols=["GAME_DATE"])
            last_date = pd.to_datetime(df["GAME_DATE"].iloc[-1])
            if last_date >= cutoff:
                active.append(d.name)
        except Exception:
            continue
    return active


def upload_full_training_bundle(bucket: str) -> None:
    """
    Upload the ENTIRE ML dataset (~2.8 GB).
    Used by GitHub Actions after full retraining.
    """
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import (
        GLOBAL_MODEL_DIR, PLAYER_DATA_DIR, AUXILIARY_DIR,
        PROCESSED_DIR, TEAM_STATS_DIR, FEATURE_COLUMNS_PATH,
    )
    
    s3 = get_r2_client()
    if not s3:
        print("R2 credentials not configured. Skipping upload.")
        return
    
    uploaded = 0
    print("Gathering files for full ML training bundle (~2.8 GB)...")
    
    upload_tasks = []
    
    # Upload everything in player_data EXCEPT the massive bloated CSV features
    for root, _, files in os.walk(PLAYER_DATA_DIR):
        for file in files:
            if file.endswith("_features.csv"):
                continue # Skip bloated redundant CSVs!
                
            local_path = Path(root) / file
            rel_path = local_path.relative_to(PLAYER_DATA_DIR)
            key = f"data/player_data/{rel_path}"
            upload_tasks.append((local_path, key))
                
    # Model artifacts
    if GLOBAL_MODEL_DIR.exists():
        for f in GLOBAL_MODEL_DIR.iterdir():
            if f.is_file():
                upload_tasks.append((f, f"artifacts/global/{f.name}"))
                
    if FEATURE_COLUMNS_PATH.exists():
        upload_tasks.append((FEATURE_COLUMNS_PATH, f"artifacts/{FEATURE_COLUMNS_PATH.name}"))

    # Delta: skip files already present in R2 at the same size. The training data is
    # append-only (new game rows grow files) and models are regenerated, so a size match
    # reliably means "unchanged". This avoids re-uploading ~48k identical files every run.
    try:
        remote_sizes = _list_remote_sizes(s3, bucket)
        deltas, skipped = [], 0
        for path, key in upload_tasks:
            try:
                if remote_sizes.get(key) == path.stat().st_size:
                    skipped += 1
                    continue
            except OSError:
                pass
            deltas.append((path, key))
        print(f"Delta: {len(deltas)} changed/new, {skipped} unchanged (skipped) of {len(upload_tasks)}.")
        upload_tasks = deltas
    except ClientError as e:
        print(f"Could not list remote for delta ({e}); uploading all files.")

    print(f"Uploading {len(upload_tasks)} files using multi-threading...")
    
    def _do_upload(task):
        path, key = task
        # Need a fresh client per thread for thread safety
        s3_thread = get_r2_client()
        return _upload_file(s3_thread, path, bucket, key)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_do_upload, task): task for task in upload_tasks}
        for future in as_completed(futures):
            if future.result():
                uploaded += 1
                if uploaded % 500 == 0:
                    print(f"  ...uploaded {uploaded} files")
                    
    print(f"Full upload complete: {uploaded} files pushed.")


def download_full_training_bundle(bucket: str) -> None:
    """Download the ENTIRE ML dataset (~2.8 GB) for GitHub Actions retraining."""
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import (
        GLOBAL_MODEL_DIR, PLAYER_DATA_DIR, AUXILIARY_DIR,
        PROCESSED_DIR, TEAM_STATS_DIR, ARTIFACTS_DIR,
    )
    
    s3 = get_r2_client()
    if not s3:
        print("R2 credentials not configured.")
        return
        
    print("Gathering remote files for full ML training bundle (~2.8 GB)...")
    prefix_to_local = {
        "artifacts/global/": GLOBAL_MODEL_DIR,
        "artifacts/": ARTIFACTS_DIR,
        "data/auxiliary/": AUXILIARY_DIR,
        "data/processed/": PROCESSED_DIR,
        "data/team_stats/": TEAM_STATS_DIR,
        "data/player_data/": PLAYER_DATA_DIR,
    }
    
    download_tasks = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket)
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                s3_key = obj['Key']
                
                local_path = None
                for prefix, local_dir in prefix_to_local.items():
                    if s3_key.startswith(prefix):
                        rel = s3_key[len(prefix):]
                        if rel:
                            local_path = local_dir / rel
                        break
                
                if not local_path or local_path.exists():
                    continue
                    
                download_tasks.append((s3_key, local_path))
    except Exception as e:
        print(f"Failed to list objects: {e}")
        return

    print(f"Downloading {len(download_tasks)} new files using multi-threading...")
    downloaded = 0
    
    def _do_download(task):
        key, path = task
        s3_thread = get_r2_client()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3_thread.download_file(bucket, key, str(path))
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_do_download, task): task for task in download_tasks}
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
                if downloaded % 500 == 0:
                    print(f"  ...downloaded {downloaded} files")
                    
    print(f"Full download complete: {downloaded} new files.")


def upload_api_bundle(bucket: str) -> None:
    """DEPRECATED: We now upload the full clean bundle."""
    upload_full_training_bundle(bucket)


def download_api_bundle(bucket: str) -> None:
    """DEPRECATED: We now download the full clean bundle."""
    download_full_training_bundle(bucket)


def upload_directory(local_dir: Path, bucket_name: str, prefix: str = ""):
    upload_full_training_bundle(bucket_name)


def download_directory(bucket_name: str, prefix: str, local_dir: Path):
    download_full_training_bundle(bucket_name)
