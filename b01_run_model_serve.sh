#!/bin/bash
torchserve \
  --start \
  --model-store model_store \
  --models mnist=mnist.mar
