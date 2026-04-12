#!/bin/bash
# Pin ROCm-related packages to prevent auto-updates that break GPU support
# Run with sudo: sudo bash scripts/pin_rocm_packages.sh

set -e

echo "=========================================="
echo "Pinning ROCm/Critical Packages"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo)"
    exit 1
fi

# Current versions
echo ""
echo "Current versions:"
echo "  Kernel: $(uname -r)"
echo "  Firmware packages:"
dpkg -l | grep -E "linux-firmware|linux-image|linux-headers" | grep -v "^rc" | head -10

echo ""
echo "=========================================="
echo "Creating package pins..."
echo "=========================================="

# Create apt preferences directory if needed
mkdir -p /etc/apt/preferences.d

# Pin linux-firmware packages to current version
cat > /etc/apt/preferences.d/pin-rocm-firmware << 'EOF'
# Pin firmware packages to prevent ROCm-breaking updates
# Stable combination: kernel 6.18.6-200 + firmware 20260110
# See: https://github.com/ROCm/ROCm/issues/

Package: linux-firmware
Pin: version 20260110*
Pin-Priority: 1001

Package: linux-firmware-*
Pin: version 20260110*
Pin-Priority: 1001

# Also block known broken versions
Package: linux-firmware
Pin: version 20251125*
Pin-Priority: -1

Package: linux-firmware-*
Pin: version 20251125*
Pin-Priority: -1
EOF

# Pin kernel packages
cat > /etc/apt/preferences.d/pin-rocm-kernel << 'EOF'
# Pin kernel to stable ROCm version
Package: linux-image-*
Pin: version 6.18.6-*
Pin-Priority: 1001

Package: linux-headers-*
Pin: version 6.18.6-*
Pin-Priority: 1001

Package: linux-modules-*
Pin: version 6.18.6-*
Pin-Priority: 1001
EOF

# Pin PyTorch/ROCm if installed via pip (we can't really pin pip, but we can document)
cat > /etc/apt/preferences.d/pin-rocm-packages << 'EOF'
# Pin ROCm system packages
Package: rocm-*
Pin: version 6.2.*
Pin-Priority: 1001

Package: hip-*
Pin: version 6.2.*
Pin-Priority: 1001

Package: miopen-*
Pin: version 6.2.*
Pin-Priority: 1001
EOF

echo "Created pin files:"
ls -la /etc/apt/preferences.d/pin-rocm-*

echo ""
echo "=========================================="
echo "Pin contents:"
echo "=========================================="
for f in /etc/apt/preferences.d/pin-rocm-*; do
    echo ""
    echo "--- $f ---"
    cat "$f"
done

echo ""
echo "=========================================="
echo "IMPORTANT NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Downgrade to stable firmware (20260110):"
echo "   sudo apt update"
echo "   sudo apt install linux-firmware=20260110* --allow-downgrades"
echo ""
echo "2. Or install specific kernel + firmware:"
echo "   sudo apt install linux-image-6.18.6-200-generic linux-headers-6.18.6-200-generic"
echo ""
echo "3. Update GRUB to boot specific kernel:"
echo "   sudo grub-set-default 'Advanced options for Ubuntu>Ubuntu, with Linux 6.18.6-200-generic'"
echo "   sudo update-grub"
echo ""
echo "4. Prevent all future updates (use with caution!):"
echo "   sudo apt-mark hold linux-firmware linux-firmware-amd-graphics linux-image-$(uname -r)"
echo ""
echo "5. To unpin later:"
echo "   sudo rm /etc/apt/preferences.d/pin-rocm-*"
echo "   sudo apt-mark unhold linux-firmware linux-firmware-amd-graphics"
echo ""
echo "=========================================="
echo "Current apt policy:"
echo "=========================================="
apt policy linux-firmware | head -5
