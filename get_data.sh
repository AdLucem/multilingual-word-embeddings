#!/bin/bash

cd data
wget -O train.txt https://s3.amazonaws.com/arrival/dictionaries/fr-en.0-5000.txt
wget -O test.txt https://s3.amazonaws.com/arrival/dictionaries/fr-en.5000-6500.txt
