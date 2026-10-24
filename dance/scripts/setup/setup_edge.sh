#!/usr/bin/env bash
# Install EDGE (Tseng et al., CVPR 2023, MIT) outside the repo for
# dance/scripts/generate/generate_motion_edge.sh, as used from Task 14 (2026-10-20) on.
#
# Usage: bash dance/scripts/setup/setup_edge.sh
#
# Result:
#   $HOME/external/EDGE               EDGE at a pinned commit (model code, MIT)
#   $HOME/external/jukemirlib         Jukebox feature helper at a pinned commit (MIT)
#   $HOME/external/jukebox_fork       Jukebox fork used by jukemirlib (OpenAI non-commercial license)
#   $HOME/external/pytorch3d_src      pytorch3d transforms/common only
#   $HOME/external/dl_edge/checkpoint.pt             EDGE checkpoint (1.1 GB, Google Drive)
#   $HOME/external/dl_jukebox/{vqvae,prior_level_2}.pth.tar   Jukebox 5B weights (10.3 GB)
#   $HOME/.venvs/edge                 uv venv, Python 3.10, torch 2.7.1 cu128
# The repo packages are not installed into the venv. edge_generate.py puts
# src/ on sys.path and only imports the numpy/scipy parts of microduck_dance.
# Needs uv, git, curl, about 13 GB of disk and 16 GB of free RAM when it runs.
# The 5B Jukebox prior is streamed to the GPU layer by layer, so 8 GB of VRAM is enough.
# SHA-256 of the weights match the ones on the Motius EDGE model card.
set -euo pipefail

UV=${UV:-$HOME/.local/bin/uv}
EXT=$HOME/external
VENV=$HOME/.venvs/edge
PY=$VENV/bin/python

EDGE_COMMIT=17c3428669ed6733edd9d8c66f7dc62060b8e46d
JUKEMIR_COMMIT=a91d87fcae0dd89085752421e794ea7e1b300735
JUKEBOX_COMMIT=6b4486164e150286d9d6944797a14b19f31912f0
P3D_COMMIT=75ebeeaea0908c5527e7b1e305fbc7681382db47
EDGE_GDRIVE_ID=1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ
EDGE_SHA=28ca4ce167bb17c36869b4d021af8762a34c6df034002f61b3bc1c1d0b1b02c7
JUKEBOX_URL=https://openaipublic.blob.core.windows.net/jukebox/models/5b
VQVAE_SHA=69745413a48e887f8a3fe91b972a6f7f434021a1ce911a99187b331eb48c059a
PRIOR_SHA=89a1dd14f5b2f9b16b3e73b53fa2138cc89fd96bb13249b4267fea471de92672

mkdir -p "$EXT" "$EXT/dl_edge" "$EXT/dl_jukebox" "$(dirname "$VENV")"

clone_at() {  # url dir commit
    [[ -d $2/.git ]] || git clone "$1" "$2"
    git -C "$2" fetch --quiet origin "$3" || true
    git -C "$2" checkout --quiet "$3"
}

# 1. Code at pinned commits.
clone_at https://github.com/Stanford-TML/EDGE "$EXT/EDGE" "$EDGE_COMMIT"
clone_at https://github.com/rodrigo-castellon/jukemirlib "$EXT/jukemirlib" "$JUKEMIR_COMMIT"
clone_at https://github.com/rodrigo-castellon/jukebox "$EXT/jukebox_fork" "$JUKEBOX_COMMIT"

# 2. venv and PyTorch 2.7.1 cu128.
[[ -x $PY ]] || "$UV" venv "$VENV" --python 3.10
"$UV" pip install --python "$PY" \
    torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$PY" numpy==1.26.4 scipy==1.15.3 librosa==0.11.0 soundfile==0.14.0 \
    einops==0.8.2 accelerate==1.8.1 p_tqdm==1.4.2 matplotlib==3.10.9 tqdm fire unidecode numba wget

# 3. jukemirlib and the Jukebox fork go on sys.path through .pth files, so their
#    pinned requirements (old tqdm, soundfile) do not downgrade the venv.
site=$("$PY" -c "import site; print(site.getsitepackages()[0])")
printf '%s\n%s\n' "$EXT/jukebox_fork" "$EXT/jukemirlib" > "$site/jukebox_src.pth"

# 4. pytorch3d: only pytorch3d.transforms is imported by EDGE, and it is pure Python.
if [[ ! -d $EXT/pytorch3d_src/.git ]]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/facebookresearch/pytorch3d "$EXT/pytorch3d_src"
fi
git -C "$EXT/pytorch3d_src" sparse-checkout set pytorch3d/transforms pytorch3d/common
git -C "$EXT/pytorch3d_src" fetch --quiet --depth 1 origin "$P3D_COMMIT"
git -C "$EXT/pytorch3d_src" checkout --quiet "$P3D_COMMIT"
echo "$EXT/pytorch3d_src" > "$site/pytorch3d_transforms_shim.pth"

# 5. Weights. Google Drive asks to confirm large downloads; the uuid form below
#    works for public files. Falls back to telling the user to download by hand.
check_sha() { echo "$2  $1" | sha256sum -c --quiet; }
ck=$EXT/dl_edge/checkpoint.pt
if [[ ! -f $ck ]]; then
    page=$(curl -fsSL "https://drive.usercontent.google.com/download?id=$EDGE_GDRIVE_ID&export=download")
    uuid=$(sed -n 's/.*name="uuid" value="\([^"]*\)".*/\1/p' <<< "$page")
    curl -fL -o "$ck" "https://drive.usercontent.google.com/download?id=$EDGE_GDRIVE_ID&export=download&confirm=t&uuid=$uuid" \
        || { echo "download https://drive.google.com/file/d/$EDGE_GDRIVE_ID to $ck by hand" >&2; exit 1; }
fi
check_sha "$ck" "$EDGE_SHA"
for f in vqvae.pth.tar:$VQVAE_SHA prior_level_2.pth.tar:$PRIOR_SHA; do
    name=${f%%:*}
    [[ -f $EXT/dl_jukebox/$name ]] || curl -fL -o "$EXT/dl_jukebox/$name" "$JUKEBOX_URL/$name"
    check_sha "$EXT/dl_jukebox/$name" "${f##*:}"
done

# 6. Check.
"$PY" -c "import torch, jukemirlib, jukebox, pytorch3d.transforms, librosa; print(torch.__version__, torch.cuda.is_available())"
