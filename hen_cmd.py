
"""
Hen-2B Model Management CLI
Complete model management system with all tiers
"""

import os
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

##########################################################################################################################
# HEN COMANDLINE TRAINER V2 - Enhanced Edition                                                                           #
# Major Updates:                                                                                                         #
# Completed the model management system                                                                                  #
# do.not.copy.please                                                                                                     #
##########################################################################################################################


# ==================== CONFIGURATION ====================
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODELS_DIR = "HenModels"

# Complete tier definitions with all metadata
ARCHETYPES = {

    "LITE": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "desc": "Fast, lightweight adapter for speed (1.5B base)",
        "vram": "6GB",
        "base_model": "lite",
        "color": "#00ffcc",
        "category": "lite"
    },
    "LITE_CODE": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "desc": "Code-focused lightweight model (1.5B base)",
        "vram": "6GB",
        "base_model": "lite",
        "color": "#00e6b8",
        "category": "lite"
    },
    

    "MINI": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "desc": "Quick test model, balanced performance (3B base)",
        "vram": "8GB",
        "base_model": "standard",
        "color": "#ffcc00",
        "category": "standard"
    },
    "MAX": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.08,
        "desc": "Production-quality general model (3B base)",
        "vram": "8GB",
        "base_model": "standard",
        "color": "#ff9900",
        "category": "standard"
    },
    "ULTRA": {
        "r": 48,
        "alpha": 96,
        "dropout": 0.1,
        "desc": "High-quality, extensive training (3B base)",
        "vram": "8GB",
        "base_model": "standard",
        "color": "#ff6600",
        "category": "standard"
    },
    "PRO": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.08,
        "desc": "Balanced mix: chat + code + logic (3B base)",
        "vram": "8GB",
        "base_model": "standard",
        "color": "#cc66ff",
        "category": "standard"
    },
    

    "CODE": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.08,
        "desc": "Python specialist with code datasets (3B base)",
        "vram": "8GB",
        "base_model": "standard",
        "color": "#00ccff",
        "category": "code"
    },
    "CODE_MAX": {
        "r": 48,
        "alpha": 96,
        "dropout": 0.1,
        "desc": "Heavy code training for complex tasks (3B base)",
        "vram": "8GB",
        "base_model": "standard",
        "color": "#0099ff",
        "category": "code"
    },
    "CODE_EXPERT": {
        "r": 64,
        "alpha": 128,
        "dropout": 0.1,
        "desc": "90% pure code training, expert level (3B base)",
        "vram": "8GB",
        "base_model": "standard",
        "color": "#0066ff",
        "category": "code"
    },
    

    "R1_MINI": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "desc": "Reasoning model with <think> tags (QwQ-32B)",
        "vram": "16GB",
        "base_model": "reasoning",
        "color": "#ff0055",
        "category": "reasoning",
        "warning": "Requires 16GB+ VRAM"
    },
    "R1_MAX": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.08,
        "desc": "Heavy reasoning for complex problems (QwQ-32B)",
        "vram": "20GB",
        "base_model": "reasoning",
        "color": "#cc0044",
        "category": "reasoning",
        "warning": "Requires 20GB+ VRAM"
    },
    "R2_DEEP": {
        "r": 48,
        "alpha": 96,
        "dropout": 0.1,
        "desc": "Math expert with deep reasoning (QwQ-32B)",
        "vram": "24GB",
        "base_model": "reasoning",
        "color": "#990033",
        "category": "reasoning",
        "warning": "Requires 24GB+ VRAM"
    },
    

    "ONI": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.08,
        "desc": "Universal: Code + Math + Chat (3B base)",
        "vram": "8GB",
        "base_model": "standard",
        "color": "#9933ff",
        "category": "specialized"
    },
    "MATH": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.08,
        "desc": "Math specialist with GSM8K focus (3B base)",
        "vram": "8GB",
        "base_model": "standard",
        "color": "#6600cc",
        "category": "specialized"
    }
}

# ==================== UTILITY FUNCTIONS ====================
def print_header():
    """Print a nice header"""
    print("\n" + "=" * 70)
    print("🐔 HEN-2B MODEL MANAGEMENT CLI")
    print("=" * 70 + "\n")

def print_colored(text, color_code):
    """Print colored text for terminal"""
    print(f"\033[{color_code}m{text}\033[0m")

def ensure_models_dir():
    """Ensure the models directory exists"""
    os.makedirs(MODELS_DIR, exist_ok=True)

# ==================== MODEL MANAGEMENT ====================
def list_models():
    """List all models in the directory"""
    print_header()
    
    if not os.path.exists(MODELS_DIR):
        print_colored(f"⚠️  Directory '{MODELS_DIR}' not found.", "93")
        ensure_models_dir()
        print(f"✅ Created directory: {MODELS_DIR}\n")
        return
    
    models = []
    for item in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(path):
            desc_path = os.path.join(path, "hen_metadata.json")
            

            if os.path.exists(desc_path):
                with open(desc_path, 'r') as f:
                    data = json.load(f)
                    models.append({
                        "name": item,
                        "tier": data.get("tier", "UNKNOWN"),
                        "status": "ready" if data.get("training_steps") else "untrained",
                        "created": data.get("created_at", ""),
                        "base": data.get("base_model", "")
                    })
            else:

                old_desc = os.path.join(path, "hen_desc.json")
                if os.path.exists(old_desc):
                    with open(old_desc, 'r') as f:
                        data = json.load(f)
                        models.append({
                            "name": item,
                            "tier": data.get("tier", "UNKNOWN"),
                            "status": data.get("status", "unknown"),
                            "created": data.get("created_at", ""),
                            "base": ""
                        })
                else:
                    models.append({
                        "name": item,
                        "tier": "UNKNOWN",
                        "status": "unknown",
                        "created": "",
                        "base": ""
                    })
    
    if not models:
        print("📭 No models found. Create one with:")
        print_colored("   python hen_cmd.py gen <name> <tier>", "96")
        print("\n")
        return
    
    print(f"📦 Found {len(models)} model(s):\n")
    print(f"{'NAME':<25} {'TIER':<15} {'STATUS':<12} {'CREATED':<20}")
    print("-" * 75)
    
    for model in models:
        status_icon = "✅" if model["status"] == "ready" else "⚠️"
        print(f"{model['name']:<25} {model['tier']:<15} {status_icon} {model['status']:<10} {model['created'][:19] if model['created'] else 'N/A':<20}")
    
    print("\n")

def create_model_slot(name, tier):
    """Create a new model slot with metadata"""
    tier = tier.upper()
    
    if tier not in ARCHETYPES:
        print_colored(f"❌ Unknown tier: {tier}", "91")
        print("\nAvailable tiers:")
        for t, config in ARCHETYPES.items():
            print(f"  • {t:<12} - {config['desc']}")
        print("\n")
        return False
    
    path = os.path.join(MODELS_DIR, name)
    
    if os.path.exists(path):
        print_colored(f"⚠️  Model '{name}' already exists!", "93")
        return False
    
    ensure_models_dir()
    os.makedirs(path, exist_ok=True)
    
    config = ARCHETYPES[tier]
    

    metadata = {
        "name": name,
        "tier": tier,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "initialized_untrained",
        "training_config": {
            "r": config["r"],
            "alpha": config["alpha"],
            "dropout": config["dropout"]
        },
        "description": config["desc"],
        "vram_requirement": config["vram"],
        "base_model_type": config["base_model"],
        "category": config["category"]
    }
    
    # Add warning if exists
    if "warning" in config:
        metadata["warning"] = config["warning"]
    
    # Save metadata
    with open(os.path.join(path, "hen_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print_colored(f"✅ Created model slot: {name}", "92")
    print(f"   Tier: {tier} ({config['desc']})")
    print(f"   VRAM: {config['vram']}")
    if "warning" in config:
        print_colored(f"   ⚠️  {config['warning']}", "93")
    print(f"\n📝 Next steps:")
    print(f"   1. Run: python hen_trainer.py")
    print(f"   2. Select tier number for {tier}")
    print(f"   3. Wait for training to complete")
    print(f"   4. Load in web interface\n")
    
    return True

def delete_model(name):
    """Delete a model and its files"""
    path = os.path.join(MODELS_DIR, name)
    
    if not os.path.exists(path):
        print_colored(f"❌ Model '{name}' does not exist.", "91")
        return False
    
    # Show model info before deletion
    desc_path = os.path.join(path, "hen_metadata.json")
    if os.path.exists(desc_path):
        with open(desc_path, 'r') as f:
            data = json.load(f)
            print(f"\n📋 Model Info:")
            print(f"   Name: {name}")
            print(f"   Tier: {data.get('tier', 'Unknown')}")
            print(f"   Created: {data.get('created_at', 'Unknown')}")
    
    confirm = input(f"\n🔥 Are you SURE you want to DELETE '{name}'? (yes/no): ")
    
    if confirm.lower() in ['yes', 'y']:
        shutil.rmtree(path)
        print_colored(f"🗑️  Deleted: {name}\n", "92")
        return True
    else:
        print("❌ Cancelled.\n")
        return False

def show_tier_info():
    """Display information about all available tiers"""
    print_header()
    print("📊 AVAILABLE MODEL TIERS\n")
    
    categories = {
        "lite": "💡 LITE TIER (Low VRAM - 1.5B)",
        "standard": "⚡ STANDARD TIER (3B Qwen)",
        "code": "💻 CODE TIER (Enhanced Datasets)",
        "reasoning": "🧠 REASONING TIER (QwQ-32B)",
        "specialized": "🌟 SPECIALIZED TIER"
    }
    
    for category, title in categories.items():
        print_colored(title, "96")
        print("-" * 70)
        
        for tier_name, config in ARCHETYPES.items():
            if config["category"] == category:
                print(f"\n{tier_name}:")
                print(f"  Description: {config['desc']}")
                print(f"  VRAM: {config['vram']}")
                print(f"  LoRA Rank: {config['r']} | Alpha: {config['alpha']}")
                if "warning" in config:
                    print_colored(f"  ⚠️  {config['warning']}", "93")
        
        print("\n")

def model_info(name):
    """Show detailed information about a specific model"""
    path = os.path.join(MODELS_DIR, name)
    
    if not os.path.exists(path):
        print_colored(f"❌ Model '{name}' not found.", "91")
        return
    
    print_header()
    print(f"📋 MODEL INFORMATION: {name}\n")
    
    # Check for metadata
    meta_path = os.path.join(path, "hen_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            data = json.load(f)
        
        print(f"Name:        {data.get('name', name)}")
        print(f"Tier:        {data.get('tier', 'Unknown')}")
        print(f"Status:      {data.get('status', 'Unknown')}")
        print(f"Created:     {data.get('created_at', 'Unknown')}")
        print(f"Description: {data.get('description', 'N/A')}")
        print(f"VRAM Req:    {data.get('vram_requirement', 'N/A')}")
        print(f"Base Model:  {data.get('base_model_type', 'N/A')}")
        
        if 'training_steps' in data:
            print(f"\nTraining Info:")
            print(f"  Steps:    {data.get('training_steps', 'N/A')}")
            print(f"  Time:     {data.get('training_time_minutes', 'N/A')} minutes")
            print(f"  Datasets: {', '.join(data.get('datasets', []))}")
        
        if data.get('capabilities'):
            print(f"\nCapabilities:")
            caps = data['capabilities']
            for key, value in caps.items():
                if value:
                    print(f"  ✓ {key.replace('_', ' ').title()}")
    else:
        print_colored("⚠️  No metadata found for this model.", "93")
    
    # Check file size
    try:
        size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        print(f"\nDisk Usage:  {size_mb:.1f} MB")
    except:
        pass
    
    print("\n")

# ==================== MAIN CLI ====================
def main():
    parser = argparse.ArgumentParser(
        description="🐔 Hen-2B Model Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hen_cmd.py list                    # List all models
  python hen_cmd.py gen my_model MINI       # Create a MINI tier model
  python hen_cmd.py info my_model           # Show model details
  python hen_cmd.py del my_model            # Delete a model
  python hen_cmd.py tiers                   # Show all available tiers
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # list: Show all models
    subparsers.add_parser("list", help="List all models")
    
    # gen: Create a new model slot
    gen_parser = subparsers.add_parser("gen", help="Create a new model slot")
    gen_parser.add_argument("name", help="Name of the model (e.g., my_first_hen)")
    gen_parser.add_argument("tier", 
                          choices=list(ARCHETYPES.keys()),
                          help="Model tier/archetype")
    
    # del: Delete a model
    del_parser = subparsers.add_parser("del", help="Delete a model")
    del_parser.add_argument("name", help="Name of the model to delete")
    
    # info: Show model details
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("name", help="Name of the model")
    
    # tiers: Show tier information
    subparsers.add_parser("tiers", help="Show all available tiers")
    
    args = parser.parse_args()
    
    # Execute commands
    if args.command == "list":
        print("HenCMD Comand Line Interface V2.0")
        list_models()
    elif args.command == "gen":
        print("HenCMD Comand Line Interface V2.0")
        create_model_slot(args.name, args.tier)
    elif args.command == "del":
        print("HenCMD Comand Line Interface V2.0")
        delete_model(args.name)
    elif args.command == "info":
        print("HenCMD Comand Line Interface V2.0")
        model_info(args.name)
    elif args.command == "tiers":
        print("HenCMD Comand Line Interface V2.0")
        show_tier_info()

if __name__ == "__main__":
    main()