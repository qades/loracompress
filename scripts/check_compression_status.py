#!/usr/bin/env python3
"""Check compression status and show high-error layers."""
import json
import os
import sys

meta_path = 'compressed_model/metadata.json'
if not os.path.exists(meta_path):
    print("No metadata found. Run 'make compress-model' first.")
    sys.exit(0)

with open(meta_path) as f:
    d = json.load(f)

print("="*60)
print("Compression Status")
print("="*60)
print(f"Layers: {d.get('layers_completed', 0)}")
print(f"Ratio: {d.get('compression_ratio', 0):.1f}x")
print(f"Original: {d.get('total_original_params', 0)/1e6:.1f}M params")
print(f"Compressed: {d.get('total_compressed_params', 0)/1e6:.1f}M params")
print()

# Error distribution
errors = []
for n, i in d.get('layer_metadata', {}).items():
    e = i.get('error')
    if e:
        errors.append((n, e, i.get('module_type', 'unknown')))

if errors:
    errors.sort(key=lambda x: x[1], reverse=True)
    
    print("High error layers (>0.30%):")
    count = 0
    for n, e, mt in errors:
        if e > 10.0:
            print(f"  {n}: {e:.3f}% ({mt})")
            count += 1
    if count == 0:
        print("  None - all layers within tolerance!")
    
    print()
    print("Error distribution:")
    buckets = {'<0.10%': 0, '0.10-0.20%': 0, '0.20-0.30%': 0, '>0.30%': 0}
    for n, e, mt in errors:
        if e < 0.10:
            buckets['<0.10%'] += 1
        elif e < 0.20:
            buckets['0.10-0.20%'] += 1
        elif e < 0.30:
            buckets['0.20-0.30%'] += 1
        else:
            buckets['>0.30%'] += 1
    
    for bucket, count in buckets.items():
        bar = "█" * (count // 2)
        print(f"  {bucket:12s}: {count:3d} {bar}")

print()
print("To improve quality, run:")
print("  make compress-resume    # Re-compress layers with error > 0.3%")
print("  make compress-improve   # Aggressive: error > 0.2%")
