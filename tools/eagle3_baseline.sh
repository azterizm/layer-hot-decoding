#!/bin/zsh
# Rebuild and (optionally) rerun the acceptance-route Stage 0 baseline:
# off-the-shelf EAGLE-3 (AngelSlim/Qwen3-4B_eagle3) on llama.cpp, next to DFlash2 + Q8_0 drafter.
#
#   tools/eagle3_baseline.sh            # prepare (download, convert, quantize) and run the timed sweep
#   tools/eagle3_baseline.sh --prepare  # prepare only
#
# Everything is pinned to what was measured on 2026-11-16 (results/20261116-144913-draft_quant).
# The timed sweep needs the charger, ~35 min, and nothing else on the GPU.
set -euo pipefail

REPO=${0:A:h:h}
LLAMA=$HOME/Code/llama.cpp
GGUF=$HOME/models/gguf
PY=$REPO/.venv/bin/python

EAGLE_REPO=AngelSlim/Qwen3-4B_eagle3
EAGLE_REV=fd331e59626c8e95c392381a16ee59d518727fbb
EAGLE_SAFETENSORS_SHA=58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e
TARGET_REPO=Qwen/Qwen3-4B                 # tokenizer + config only, for the converter
TARGET_REV=1cfa9a7208912126459214e8b04321603b3df60c
LLAMA_COMMIT=e3483e40b02326383a54a23ae71f2bfa9f81181a
GGUF_BF16_SHA=98fe2040a99b973baa6a543c4b30e8d8f3bfda861b80f5baf3a6d796e0835048
GGUF_Q8_SHA=5e8d1b08bb64f07b8776b039ff9ac7b18773dc2442f1cd37deaab541b2993c73

say() { print -P "%B[eagle3-baseline]%b $*"; }

# 0. prerequisites -----------------------------------------------------------------------------
[[ -x $PY ]] || { say "missing $PY (create the main venv first: see pyproject.toml)"; exit 1; }
[[ -d $LLAMA/.git ]] || { say "missing llama.cpp at $LLAMA"; exit 1; }
have=$(git -C $LLAMA rev-parse HEAD)
if [[ $have != $LLAMA_COMMIT ]]; then
  say "WARNING: llama.cpp is at ${have[1,9]}, measurements used ${LLAMA_COMMIT[1,9]}."
  say "         for identical results: git -C $LLAMA checkout $LLAMA_COMMIT && rebuild"
fi
for t in llama-server llama-quantize; do
  [[ -x $LLAMA/build/bin/$t ]] || cmake --build $LLAMA/build --config Release -j 6 --target $t
done
for f in Qwen3-4B-Q4_0.gguf Qwen3-4B-DFlash2-Q8_0.gguf; do
  [[ -f $GGUF/$f ]] || say "WARNING: $GGUF/$f missing (needed for the DFlash2 comparison arm)"
done

# 1. download (pinned revisions, retried; plain HTTP because Xet stalls on this network) -------
export HF_HUB_DISABLE_XET=1
dl() {  # repo revision [patterns...]
  local repo=$1 rev=$2; shift 2
  until $PY - $repo $rev "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, rev, pats = sys.argv[1], sys.argv[2], sys.argv[3:] or None
print(snapshot_download(repo, revision=rev, allow_patterns=pats, etag_timeout=30))
EOF
  do say "download of $repo failed, retrying in 10 s"; sleep 10; done
}
EAGLE_DIR=$(dl $EAGLE_REPO $EAGLE_REV | tail -1)
TARGET_DIR=$(dl $TARGET_REPO $TARGET_REV "*.json" "*.txt" "tokenizer*" "vocab.json" "merges.txt" | tail -1)
got=$(shasum -a 256 $EAGLE_DIR/model.safetensors | cut -d' ' -f1)
[[ $got == $EAGLE_SAFETENSORS_SHA ]] || { say "checksum mismatch for $EAGLE_REPO model.safetensors"; exit 1; }
say "downloaded and verified $EAGLE_REPO@${EAGLE_REV[1,9]}"

# 2. converter venv (llama.cpp's convert requirements) ----------------------------------------
CV=$REPO/.venv-convert
if [[ ! -x $CV/bin/python ]]; then
  uv venv $CV --python 3.12 -q
  uv pip install --python $CV/bin/python "torch==2.11.0" "transformers==4.57.6" "numpy~=2.2.6" \
    "sentencepiece>=0.1.98,<0.3.0" "protobuf>=4.21.0,<5.0.0" safetensors -e $LLAMA/gguf-py \
    --extra-index-url https://download.pytorch.org/whl/cpu
fi

# 3. convert + quantize -------------------------------------------------------------------------
mkdir -p $GGUF
(cd $LLAMA && $CV/bin/python convert_hf_to_gguf.py $EAGLE_DIR --target-model-dir $TARGET_DIR \
   --outtype bf16 --outfile $GGUF/Qwen3-4B-eagle3.gguf 2>&1 | grep -E "EAGLE-3|successfully|Error" || true)
$LLAMA/build/bin/llama-quantize $GGUF/Qwen3-4B-eagle3.gguf $GGUF/Qwen3-4B-eagle3-Q8_0.gguf Q8_0 >/dev/null 2>&1
for pair in "Qwen3-4B-eagle3.gguf:$GGUF_BF16_SHA" "Qwen3-4B-eagle3-Q8_0.gguf:$GGUF_Q8_SHA"; do
  f=${pair%%:*}; want=${pair##*:}
  got=$(shasum -a 256 $GGUF/$f | cut -d' ' -f1)
  if [[ $got == $want ]]; then say "$f identical to the measured file"
  else say "WARNING: $f differs from the measured file (different llama.cpp/converter version?)"; fi
done

[[ ${1:-} == --prepare ]] && { say "prepared; run without --prepare for the timed sweep"; exit 0; }

# 4. timed sweep (same variants as results/20261116-144913-draft_quant) ----------------------
pmset -g batt | head -1 | grep -q "AC Power" || { say "not on the charger: timed rounds would be discarded"; exit 1; }
cd $REPO
$PY -u -m bench.draft_quant --variants llama_dflash2_q8 llama_eagle3_bf16_n3 llama_eagle3_bf16_n5 \
  llama_eagle3_bf16_n7 llama_eagle3_q8_n5 --warmup-s 60 --cooldown 60
