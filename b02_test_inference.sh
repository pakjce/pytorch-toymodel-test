#!/usr/bin/env bash
curl -X POST http://127.0.0.1:8080/predictions/mnist -T ./example_images/${1}.png
