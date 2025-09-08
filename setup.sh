#!/bin/bash
set -e

echo "🚀 Setting up Swin-UNet Segmentation Framework..."
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}UV not found. Installing UV...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc || source ~/.zshrc || true
    echo -e "${GREEN}✅ UV installed successfully${NC}"
else
    echo -e "${GREEN}✅ UV found${NC}"
fi

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
uv sync

# Create necessary directories
echo -e "${BLUE}� Creating directories...${NC}"
mkdir -p data/cache
mkdir -p logs/sam
mkdir -p logs/public
mkdir -p checkpoints
mkdir -p predictions

echo -e "${GREEN}✅ Directories created${NC}"

# Run tests to verify installation
echo -e "${BLUE}🧪 Running tests to verify installation...${NC}"
if uv run python -m pytest tests/ -v --tb=short; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Some tests failed. Please check the installation.${NC}"
    exit 1
fi

echo
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "  ${BLUE}For SAM Acoustic Microscopy:${NC}"
echo "    1. Place your data in data/sam_acoustic/"
echo "    2. Run: uv run python src/data/prepare.py data/sam_acoustic configs/config_sam.yaml"
echo "    3. Train: uv run python src/train.py --config configs/config_sam.yaml"
echo
echo "  ${BLUE}For Public Dataset Demo:${NC}"
echo "    1. Run: uv run python src/data/download.py configs/config_public.yaml"
echo "    2. Prepare: uv run python src/data/prepare.py data/road_cracks configs/config_public.yaml"
echo "    3. Train: uv run python src/train.py --config configs/config_public.yaml"
echo
echo -e "${GREEN}Happy segmenting! 🔬✨${NC}"
