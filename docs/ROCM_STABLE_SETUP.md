# ROCm Stable Setup for AMD Strix Halo (gfx1151)

## Proven Working Combination

Based on community reports and testing:

| Component | Stable Version | Notes |
|-----------|---------------|-------|
| Kernel | `6.18.6-200` | 6.18+ required for XNACK support |
| Firmware | `20260110` | AVOID 20251125 - known broken |
| ROCm | `6.2.x` | Via amdgpu-install or docker |
| PyTorch | `2.12.0+rocm7.13` | From AMD gfx1151 nightly repo |

## Current Status Check

```bash
# Check kernel
uname -r
# Should be: 6.18.6-200 or newer (6.19+ also works)

# Check firmware
dmesg | grep -i firmware | grep -i amdgpu
# Look for: 20260110 (good) or 20251125 (BAD)

# Check ROCm
dpkg -l | grep -E "rocm|hip|amdgpu"

# Check PyTorch
python3 -c "import torch; print(torch.__version__)"
# Should contain: rocm
```

## Installation Steps

### 1. Pin Firmware to Stable Version

Prevent apt from auto-updating to broken firmware:

```bash
sudo bash scripts/pin_rocm_packages.sh
```

### 2. Downgrade Firmware (if needed)

If you have firmware 20251125 or 20260319:

```bash
# Download specific firmware version
# Note: This may not be in standard repos, may need manual download

# Option A: From Ubuntu archives
wget http://archive.ubuntu.com/ubuntu/pool/main/l/linux-firmware/linux-firmware_20260110_all.deb
sudo dpkg -i linux-firmware_20260110_all.deb
sudo apt-mark hold linux-firmware

# Option B: Use older kernel that bundles compatible firmware
sudo apt install linux-image-6.18.6-200-generic
```

### 3. Kernel Boot Parameters

Edit `/etc/default/grub`:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856"
```

Apply:
```bash
sudo update-grub
sudo reboot
```

### 4. Environment Variables

Add to `~/.bashrc` or run before PyTorch:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_XNACK=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_ENABLE_SDMA=0
```

### 5. PyTorch Installation

```bash
# Remove old PyTorch
pip uninstall torch torchvision torchaudio

# Install AMD gfx1151-specific build
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchvision torchaudio
```

## Verification

```bash
make fix-gpu
```

Should show:
- ✓ Kernel 6.18+
- ✓ Firmware 20260110 (or at least not 20251125)
- ✓ GPU tests passing

## Troubleshooting

### GPU Still Hangs

1. **Check XNACK is enabled:**
   ```bash
   dmesg | grep -i xnack
   # Should show: "amdgpu: XNACK enabled"
   ```

2. **Check firmware loaded correctly:**
   ```bash
   dmesg | grep -i "amdgpu.*firmware"
   ls /lib/firmware/amdgpu/ | grep -i gfx1151
   ```

3. **Try disabling GPU power management:**
   Add to kernel params: `amdgpu.runpm=0`

4. **Force PCIe gen3:**
   Add to kernel params: `amdgpu.pcie_gen3=1`

### Firmware Too New (20260319)

The 20260319 firmware is very new and may have issues. To downgrade:

```bash
# Hold current packages
sudo apt-mark hold linux-firmware linux-firmware-amd-graphics

# If you have a backup of 20260110:
sudo apt install linux-firmware=20260110 --allow-downgrades

# Otherwise, may need to wait for ROCm 6.3+ to support newer firmware
```

### Alternative: Use Docker

If system-level ROCm is problematic, use AMD's Docker image:

```bash
docker run -it --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch2.4
```

## Package Pinning Explained

The `pin_rocm_packages.sh` script creates apt preferences that:

1. **Give priority 1001** to known-good versions (will install/keep these)
2. **Give priority -1** to known-bad versions (will never install)
3. **Hold packages** via `apt-mark hold` to prevent any updates

To unpin later:
```bash
sudo rm /etc/apt/preferences.d/pin-rocm-*
sudo apt-mark unhold linux-firmware linux-firmware-amd-graphics
```

## References

- ROCm gfx1151 support: https://github.com/ROCm/ROCm/issues/3247
- AMD Strix Halo issues: https://github.com/ROCm/ROCm/issues/3251
- Firmware 20251125 bug: https://gitlab.freedesktop.org/drm/amd/-/issues/3580
