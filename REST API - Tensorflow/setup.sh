#!/bin/bash

# Download miniconda

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

echo "Download complete."

echo "Installing Miniconda"

sh Miniconda3-latest-Linux-x86_64.sh

echo "Installation Done... Setting Up environment"

conda create --name challenge

source activate challenge

echo "Installing Libraries..."


#pip install sklearn && pip install keras && pip install tensorflow && pip install flask && pip install pillow && pip install psutil
cd /c

echo "Starting Server..."

echo "For Prediction API run: \"curl -X POST -F \"image=@\sample\sample.png\" http://127.0.0.1:5000/predict"
echo "For Batch Train API run: \"curl -X POST -F \"zip=@\sample\test.zip\" http://127.0.0.1:5000/batch_train"

python server.py


