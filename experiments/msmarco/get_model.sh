#!/usr/bin/env bash
# 用 curl 下载 BGE-small-en-v1.5 到本地目录（服务器上 Python 的 HTTP 客户端走 IPv6 会报 Network is unreachable，curl 可自动回退 IPv4）。
set -euo pipefail
DST=${1:-/path/to/workspace/data_msmarco/bge-small-en-v1.5}
BASE=https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main
mkdir -p $DST/1_Pooling
for f in config.json config_sentence_transformers.json modules.json sentence_bert_config.json special_tokens_map.json \
         tokenizer.json tokenizer_config.json vocab.txt 1_Pooling/config.json model.safetensors; do
  [ -s $DST/$f ] || curl -sSL --retry 5 -o $DST/$f $BASE/$f
done
ls -la $DST $DST/1_Pooling
