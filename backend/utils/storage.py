import os
import boto3
from pathlib import Path
from botocore.exceptions import ClientError
from botocore.client import Config

def get_r2_client():
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    
    if not all([account_id, access_key, secret_key]):
        return None
        
    return boto3.client(
        's3',
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )

def upload_directory(local_dir: Path, bucket_name: str, prefix: str = ""):
    s3 = get_r2_client()
    if not s3:
        print("S3/R2 credentials not found, skipping upload.")
        return
        
    if not local_dir.exists():
        print(f"Directory {local_dir} does not exist, skipping upload.")
        return
        
    for root, _, files in os.walk(local_dir):
        for file in files:
            # Skip hidden files and large sqlite files to save time if needed
            if file.startswith('.'):
                continue
                
            local_path = Path(root) / file
            rel_path = local_path.relative_to(local_dir)
            s3_key = f"{prefix}/{rel_path}".lstrip("/")
            
            print(f"Uploading {local_path} to {bucket_name}/{s3_key}")
            try:
                s3.upload_file(str(local_path), bucket_name, s3_key)
            except ClientError as e:
                print(f"Upload failed: {e}")

def download_directory(bucket_name: str, prefix: str, local_dir: Path):
    s3 = get_r2_client()
    if not s3:
        print("S3/R2 credentials not found, skipping download.")
        return
        
    print(f"Downloading from {bucket_name}/{prefix} to {local_dir}")
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                s3_key = obj['Key']
                rel_path = s3_key[len(prefix):].lstrip("/")
                if not rel_path:
                    continue
                    
                local_path = local_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Basic caching: only download if not exists
                # In a robust production environment, you'd check ETag/MD5 hashes
                if not local_path.exists():
                    print(f"Downloading {s3_key} to {local_path}")
                    s3.download_file(bucket_name, s3_key, str(local_path))
    except ClientError as e:
        print(f"Download failed: {e}")
