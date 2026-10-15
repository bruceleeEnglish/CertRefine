#!/usr/bin/env bash
# 多连接分段下载（服务器到 HF 单连接只有 ~0.5 MB/s）。用法: dl_ranges.sh <url> <out> <size_bytes> [parts=16]
set -euo pipefail
URL=$1; OUT=$2; SIZE=$3; N=${4:-16}
TMP=$OUT.parts; mkdir -p $TMP
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N - 1))); do
  S=$(( i * CH )); E=$(( S + CH - 1 )); [ $E -ge $SIZE ] && E=$((SIZE - 1))
  WANT=$(( E - S + 1 ))
  ( for try in 1 2 3 4 5 6 7 8; do
      HAVE=$(stat -c %s $TMP/$i 2>/dev/null || echo 0)
      [ "$HAVE" -ge "$WANT" ] && break
      curl -sSL --retry 5 -r $((S + HAVE))-$E $URL >> $TMP/$i || sleep 5
    done ) &
done
wait
for i in $(seq 0 $((N - 1))); do cat $TMP/$i; done > $OUT
ls -la $OUT
[ "$(stat -c %s $OUT)" -eq "$SIZE" ] && rm -rf $TMP && echo DL_OK || echo DL_SIZE_MISMATCH
