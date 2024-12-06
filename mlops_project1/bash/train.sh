#!/bin/bash

poetry run python mlops_labs2024/modeling/train.py -d data/processed/processed.csv -p params.yml
