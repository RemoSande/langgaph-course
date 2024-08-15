import boto3
from botocore.exceptions import ClientError
import os
from typing import List

def list_s3_files(bucket_name: str, folder_path: str) -> List[str]:
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
        return [item['Key'] for item in response.get('Contents', []) if item['Key'].lower().endswith('.pdf')]
    except ClientError as e:
        print(f"An error occurred: {e}")
        return []

def download_from_s3(bucket_name: str, file_key: str) -> str:
    s3 = boto3.client('s3')
    local_file_path = os.path.join('/tmp', os.path.basename(file_key))
    try:
        s3.download_file(bucket_name, file_key, local_file_path)
        return local_file_path
    except ClientError as e:
        print(f"An error occurred: {e}")
        return None
