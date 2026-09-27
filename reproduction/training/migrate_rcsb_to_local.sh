#!/usr/bin/env bash
set -euo pipefail

SRC_DATA=/bigdat2/user/sunxc/rcsb_data/rcsb
SRC_TRAINING=/bigdat2/user/sunxc/esmdynamic_training/01_rcsb_dynamic
DEST_ROOT=/data/user/sunxc/local_esmdynamic
DEST_DATA="$DEST_ROOT/rcsb_data/rcsb"
DEST_TRAINING="$DEST_ROOT/training/01_rcsb_dynamic"

for src in "$SRC_DATA" "$SRC_TRAINING"; do
    if ! timeout 10s test -d "$src"; then
        echo "Source is unavailable: $src" >&2
        echo "Check /bigdat2 mount before retrying." >&2
        exit 2
    fi
done

mkdir -p "$DEST_DATA" "$DEST_TRAINING"

echo "Copying RCSB dataset..."
rsync -a --info=progress2 --partial "$SRC_DATA/" "$DEST_DATA/"

echo "Copying epoch-1 checkpoints and metadata..."
rsync -a --info=progress2 --partial \
    --include='checkpoint_*.pt' \
    --include='heads_*.pt' \
    --include='history.csv' \
    --include='run_metadata*.json' \
    --include='timing_summary.json' \
    --exclude='*' \
    "$SRC_TRAINING/" "$DEST_TRAINING/"

echo "Migration complete. Destination: $DEST_ROOT"
du -sh "$DEST_DATA" "$DEST_TRAINING"
