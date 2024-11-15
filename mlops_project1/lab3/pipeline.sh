#!/bin/bash

# 1. Отправка исходного файла в S3
bash lab3/bashscript/upload.sh

# 2. Загрузка исходного файла из S3
bash lab3/bashscript/download.sh

# 3. Обработка исходного файла
bash lab3/bashscript/process.sh

# 4. Загрузка обработанного файла в S3
bash lab3/bashscript/upload_process.sh
