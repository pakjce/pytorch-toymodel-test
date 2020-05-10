#!/bin/bash
torch-model-archiver \
  --model-name mnist \
  --version 1.0 \
  --model-file prepare_model/mnist_model.py \
  --serialized-file model/mnist_cnn.pt \
  --handler model_serve_handler/mnist_handler.py \
  --export-path model_store/ \
  -f
