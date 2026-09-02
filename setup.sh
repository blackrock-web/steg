#!/bin/bash

# Setup script for SecureStegVault

echo "=========================================="
echo " Setting up SecureStegVault Dependencies  "
echo "=========================================="

echo "[1/2] Installing Node.js dependencies..."
npm install

echo "[2/2] Installing Python dependencies (if needed for benchmark runner)..."
# Using --break-system-packages for linux environments where pip warns about system-wide installs,
# though normally a virtual environment is recommended.
pip3 install --break-system-packages numpy onnxruntime 2>/dev/null || pip3 install numpy onnxruntime

echo "=========================================="
echo " Setup Complete! You can now run start.sh "
echo "=========================================="
