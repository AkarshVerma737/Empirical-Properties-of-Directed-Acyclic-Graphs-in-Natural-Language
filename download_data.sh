#!/bin/bash
# download_data.sh
# Downloads SUD treebanks (v2.17) from grew.fr for LE2.
# Each treebank is a .tgz archive — this script downloads and extracts them.
# Run from the le2/ project root: bash download_data.sh

mkdir -p data tmp_extract

BASE="https://grew.fr/download/SUD_2.17"

# List of SUD treebank names to download
TREEBANKS=(
    "SUD_English-EWT"
    "SUD_Hindi-HDTB"
    "SUD_French-GSD"
    "SUD_German-GSD"
    "SUD_Spanish-GSD"
    "SUD_Chinese-GSD"
    "SUD_Arabic-PADT"
    "SUD_Russian-SynTagRus"
    "SUD_Japanese-GSD"
    "SUD_Turkish-IMST"
    "SUD_Basque-BDT"
)

for TREEBANK in "${TREEBANKS[@]}"; do
    TGZ_URL="$BASE/$TREEBANK.tgz"
    TGZ_FILE="tmp_extract/$TREEBANK.tgz"

    echo "Downloading $TREEBANK ..."
    curl -fsSL "$TGZ_URL" -o "$TGZ_FILE"

    if [ $? -ne 0 ]; then
        echo "  FAILED: $TGZ_URL"
        continue
    fi

    echo "  Extracting ..."
    tar -xzf "$TGZ_FILE" -C tmp_extract/

    # Find the train conllu file inside the extracted folder
    TRAIN_FILE=$(find "tmp_extract/$TREEBANK/" -name "*-train.conllu" | head -1)

    if [ -z "$TRAIN_FILE" ]; then
        echo "  WARNING: No train file found in $TREEBANK"
        # List what's actually there
        ls "tmp_extract/$TREEBANK/" 2>/dev/null || true
    else
        DEST="data/$(basename $TRAIN_FILE)"
        cp "$TRAIN_FILE" "$DEST"
        echo "  Saved: $DEST"
    fi

    # Cleanup extracted folder to save space
    rm -rf "tmp_extract/$TREEBANK/" "$TGZ_FILE"
done

rmdir tmp_extract 2>/dev/null || true

echo ""
echo "Done. Files in ./data/:"
ls -lh data/
