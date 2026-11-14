#!/bin/sh
set -eu

model=/models/Qwen3-Embedding-8B-Q4_K_M.gguf
part=/models/Qwen3-Embedding-8B-Q4_K_M.gguf.part
marker=/models/Qwen3-Embedding-8B-Q4_K_M.gguf.verified
sha256=3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85
url=https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/69d0e58a13e463cd99a9b83e3f5fee7c10265fab/Qwen3-Embedding-8B-Q4_K_M.gguf

# "<sha256> <size> <mtime>" is written after a full check; an unchanged file is not re-read
# on every start (4.7 GB of disk reads and page cache).
write_marker() {
    echo "$sha256 $(stat -c '%s %Y' "$model")" > "$marker"
}

if [ -f "$model" ]; then
    if [ -f "$marker" ] && [ "$(cat "$marker")" = "$sha256 $(stat -c '%s %Y' "$model")" ]; then
        echo "model verified earlier: $model"
        exit 0
    fi
    echo "$sha256  $model" | sha256sum -c -
    write_marker
    exit 0
fi

curl --fail --location --retry 4 --retry-delay 3 --continue-at - --output "$part" "$url"
echo "$sha256  $part" | sha256sum -c -
mv "$part" "$model"
write_marker
