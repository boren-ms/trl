#!/bin/bash

echo "Listing GPU usage:"
# List processes using NVIDIA devices
nvidia-smi |grep Default
echo "Done"