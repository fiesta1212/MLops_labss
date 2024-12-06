import argparse
from dotenv import load_dotenv

from mlops_labs2024.create_bucket import create_bucket, create_client

load_dotenv()

argparser = argparse.ArgumentParser()
argparser.add_argument("-b", "--bucket", required=True, help="S3 bucket name")
argparser.add_argument("-f", "--file_path", required=True, help="File path to upload")
argparser.add_argument("-d", "--data_path", required=True, help="Data file name in S3")
argparser.add_argument(
    "-c",
    "--create_bucket",
    action="store_true",
    help="Create the bucket if it does not exist",
)


def upload_file(bucket, object_name, file_path):
    s3 = create_client()
    s3.upload_file(file_path, bucket, object_name)
    print(f"File '{object_name}' uploaded to bucket '{bucket}'.")


if __name__ == "__main__":
    args = argparser.parse_args()

    if args.create_bucket:
        create_bucket(args.bucket)

    upload_file(args.bucket, args.data_path, args.file_path)
