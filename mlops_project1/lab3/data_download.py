from dotenv import load_dotenv
import argparse

from lab3.bucket_create import create_client

load_dotenv()

argparser = argparse.ArgumentParser()
argparser.add_argument("-b", "--bucket", required=True, help="S3 bucket name")
argparser.add_argument(
    "-i", "--input_path", required=True, help="Path of the input file in S3"
)
argparser.add_argument(
    "-o", "--output_path", required=True, help="Path to save file in S3"
)


def download_file(bucket, object_name, file_name):
    s3 = create_client()
    s3.download_file(bucket, object_name, file_name)


if __name__ == "__main__":
    args = argparser.parse_args()
    bucket = args.bucket
    object_name = args.input_path
    file_name = args.output_path
    download_file(bucket, object_name, file_name)
