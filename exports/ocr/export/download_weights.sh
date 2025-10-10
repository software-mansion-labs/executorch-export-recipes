#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download weights into the export directory
cd "$SCRIPT_DIR"

# CRAFT detector
wget -O craft_mlt_25k.zip https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip
unzip -o craft_mlt_25k.zip
rm craft_mlt_25k.zip

# ENGLISH CRNN
wget -O english_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip
unzip -o english_g2.zip
rm english_g2.zip

# LATIN CRNN
wget -O latin_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip
unzip -o latin_g2.zip
rm latin_g2.zip

# CHINESE CRNN
wget -O zh_sim_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip
unzip -o zh_sim_g2.zip
rm zh_sim_g2.zip

# JAPANESE CRNN
wget -O japanese_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/japanese_g2.zip
unzip -o japanese_g2.zip
rm japanese_g2.zip

# KOREAN CRNN
wget -O korean_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.zip
unzip -o korean_g2.zip
rm korean_g2.zip

# TELUGU CRNN
wget -O telugu.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.2/telugu.zip
unzip -o telugu.zip
rm telugu.zip

# KANNADA CRNN
wget -O kannada.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.2/kannada.zip
unzip -o kannada.zip
rm kannada.zip

# CYRILLIC CRNN
wget -O cyrillic_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.1/cyrillic_g2.zip
unzip -o cyrillic_g2.zip
rm cyrillic_g2.zip

echo "All weights downloaded successfully to $SCRIPT_DIR"