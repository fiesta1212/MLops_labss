import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

load_dotenv()


def create_client():
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id=os.getenv("USER_NAME"),
        aws_secret_access_key=os.getenv("PASSWORD"),
    )
    return s3


def create_bucket(bucket_name):
    s3 = create_client()
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created successfully.")
    except ClientError as e:
        print(f"Bucket '{bucket_name}' already exists: {e}")


if __name__ == "__main__":
    bucket_name = "data"
    create_bucket(bucket_name)
