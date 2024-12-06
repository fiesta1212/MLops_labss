#!/bin/bash
poetry install
docker-compose up --build -d


bash bash/upload.sh

bash bash/download.sh

bash bash/process.sh

bash bash/upload_process.sh

bash bash/train.sh
