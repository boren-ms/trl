#!/bin/sh
# This script lists the process IDs of processes using NVIDIA devices.

echo "list processes using NVIDIA devices"
lsof /dev/nvidia* | awk '{print $2}' | grep -E '^[0-9]+$' | sort -u

echo "killing processes using NVIDIA devices"
lsof /dev/nvidia* | awk '{print $2}' | grep -E '^[0-9]+$' | sort -u | xargs kill -9

echo "Check process again."
lsof /dev/nvidia* | awk '{print $2}' | grep -E '^[0-9]+$' | sort -u
echo "GPUs released."