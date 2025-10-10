# OCR Model Export Utilities

Export utilities for converting EasyOCR models to ExecuTorch format with XNNPACK backend optimization.

## Overview

This directory contains scripts to export CRAFT text detection and CRNN text recognition models for mobile deployment.

## Scripts

### `download_weights.sh`
Downloads pre-trained weights for all supported models from the EasyOCR repository.

### `craft.py`
Exports CRAFT text detection models in three different input sizes:
- **1280x1280** - Full resolution
- **1280x320** - Vertical text detection
- **800x800** - Standard resolution

Generates both quantized and non-quantized versions for each size.

### `crnn.py`
Exports CRNN text recognition models for 8 languages across 4 input widths (64, 128, 256, 512):

| Language | Classes | Weight File |
|----------|---------|-------------|
| English | 97 | `english_g2.pth` |
| Latin | 352 | `latin_g2.pth` |
| Japanese | 2,215 | `japanese_g2.pth` |
| Chinese (Simplified) | 6,719 | `zh_sim_g2.pth` |
| Korean | 1,009 | `korean_g2.pth` |
| Telugu | 166 | `telugu.pth` |
| Kannada | 168 | `kannada.pth` |
| Cyrillic | 208 | `cyrillic.pth` |

## Usage

1. Download weights:
   ```bash
   bash download_weights.sh
   ```

2. Export CRAFT models:
   ```bash
   python craft.py
   ```

3. Export CRNN models(This can run for a long time):
   ```bash
   python crnn.py
   ```

## Output

- CRAFT: `xnnpack_craft_{width}.pte` and `xnnpack_craft_{width}_quantized.pte`
- CRNN: `xnnpack_crnn_{language}_{width}.pte`

## Requirements

- PyTorch
- ExecuTorch
- OpenCV (for CRAFT calibration)
- NumPy
