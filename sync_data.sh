#!/bin/bash
# add azcopy to path
export PATH=$PATH:/root/data/tools/

if ! grep -Fxq "export PATH=\$PATH:/root/data/tools/" ~/.bashrc; then
    echo 'export PATH=$PATH:/root/data/tools/' >> ~/.bashrc
fi

source ~/.bashrc

bash ./SimpleSciClone.sh orngtransfer orngscuscresco data /speech/am_data/en/human_caption_v2/FY22_AdjustBoundary_BiasLM/ChunkFiles