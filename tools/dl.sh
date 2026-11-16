#!/bin/zsh
# robust downloads: retry until complete
PY=${0:A:h:h}/.venv/bin/python
export HF_HUB_DISABLE_XET=1
hf_get() {  # repo [allow patterns...]
  local repo=$1; shift
  until $PY -c "
import sys
from huggingface_hub import snapshot_download as s
pats = sys.argv[2:] or None
print(s(sys.argv[1], allow_patterns=pats, etag_timeout=30))
" $repo "$@" 2>&1 | grep -v -i "warn" | tail -1 | grep -q snapshots; do echo "retry $repo $(date +%T)"; sleep 5; done
  echo "OK $repo $(date +%T)"
}
gguf_get() {
  local url=https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_0.gguf
  local out=$HOME/models/gguf/Qwen3-4B-Q4_0.gguf
  local want=$(curl -sIL $url | grep -i '^content-length' | tail -1 | tr -dc 0-9)
  until [ "$(stat -f %z $out 2>/dev/null)" = "$want" ]; do
    curl -sSL -C - --speed-limit 20000 --speed-time 60 -o $out $url; sleep 3
  done
  echo "OK gguf 4B $(stat -f %z $out) == $want $(date +%T)"
}
case $1 in
  mlx4b) hf_get mlx-community/Qwen3-4B-4bit ;;
  gguf4b) gguf_get ;;
  drafts)
    hf_get Qwen/Qwen3-4B "*.json" "*.txt" "tokenizer*" "vocab.json" "merges.txt"
    hf_get z-lab/Qwen3-4B-DFlash-b16 "*.json" "*.py" "*.safetensors"
    hf_get mgoin/Qwen3-4B-speculator.dflash2 "*.json" "*.py" "*.safetensors" "SHA256SUMS" ;;
esac
