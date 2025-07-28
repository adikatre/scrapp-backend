#!/bin/bash

set -e

echo "downloading dataset"
echo "==================="

if ! command -v kaggle &> /dev/null; then
    echo "kaggle not installed. install requirements.txt"
    exit 1
fi

# Check if kaggle credentials are configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo "Please set up your Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Click 'Create New API Token' to download kaggle.json"
    echo "3. Place kaggle.json in ~/.kaggle/ directory"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi


DATASET_DIR="data/taco_vencer"
DATASET_NAME="vencerlanz09/taco-dataset-yolo-format"

rm -rf $DATASET_DIR
mkdir $DATASET_DIR

echo "⬇Downloading TACO dataset from Kaggle..."
echo "Dataset: $DATASET_NAME"

cd "$DATASET_DIR"
kaggle datasets download -d "$DATASET_NAME"

# Extract the downloaded zip file
echo "📦 Extracting dataset..."
if [ -f "taco-dataset-yolo-format.zip" ]; then
    unzip -q "taco-dataset-yolo-format.zip"
    echo "✅ Dataset extracted successfully!"
    
    # Clean up zip file
    rm "taco-dataset-yolo-format.zip"
    echo "🧹 Cleaned up zip file"
else
    echo "❌ Downloaded zip file not found!"
    exit 1
fi

echo "✅ TACO dataset download completed!"
echo "📍 Dataset location: $(pwd)/$DATASET_DIR"

rm -rf images
rm -rf labels
mkdir images
mkdir labels

cp -r ./test/images ./images/test
cp -r ./train/images ./images/train
cp -r ./valid/images ./images/valid

cp -r ./test/labels ./labels/test
cp -r ./train/labels ./labels/train
cp -r ./valid/labels ./labels/valid

rm -rf train valid test