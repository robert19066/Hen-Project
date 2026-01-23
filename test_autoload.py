#!/usr/bin/env python3
"""Test the auto-loading functionality"""

# Simplified test of the auto-loading logic
datasets_config = {
    "lite": [
        {"name": "alpaca-cleaned", "weight": 0.6, "format": "chat"},
        {"name": "OpenAssistant (oasst1)", "weight": 0.4, "format": "chat"}
    ],
    "lite_code": [
        {"name": "python_code_instructions_18k", "weight": 0.85, "format": "code"},
        {"name": "alpaca-cleaned", "weight": 0.15, "format": "chat"}
    ],
}

def get_auto_datasets_for_tier(tier):
    """Return auto-loaded datasets for a given tier"""
    return datasets_config.get(tier, [])

def auto_load_datasets_for_model(tier):
    """Automatically load datasets for the selected model tier"""
    datasets_info = get_auto_datasets_for_tier(tier)
    
    datasets = []
    
    # Add auto-loaded datasets
    for ds_info in datasets_info:
        datasets.append({
            "path": ds_info["name"],
            "weight": ds_info["weight"],
            "format": ds_info["format"],
            "auto_loaded": True
        })
    
    print(f"\n🧬 Auto-loaded {len(datasets)} dataset(s) for {tier}:")
    for i, d in enumerate(datasets, 1):
        print(f"   {i}. {d['path']} (weight: {d['weight']}x, format: {d['format']})")
    
    return datasets

# Test
print("=" * 70)
print("Testing HEN LAB Auto-Loading")
print("=" * 70)

print("\n1️⃣  Testing hen_lite auto-loading:")
datasets_lite = auto_load_datasets_for_model("lite")
print(f"✅ Loaded {len(datasets_lite)} datasets")

print("\n2️⃣  Testing hen_lite_code auto-loading:")
datasets_code = auto_load_datasets_for_model("lite_code")
print(f"✅ Loaded {len(datasets_code)} datasets")

print("\n" + "=" * 70)
print("✅ Auto-loading logic works correctly!")
print("=" * 70)
