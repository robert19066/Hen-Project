import os
import json
import shutil
import argparse
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODELS_DIR = "HenModels"

# Define what makes each tier unique
ARCHETYPES = {
    "LITE": {
        "r": 8, "alpha": 16, "dropout": 0.05, 
        "desc": "Fast, lightweight adapter for speed.", 
        "color": "#00ffcc"
    },
    "MINI": {
        "r": 16, "alpha": 32, "dropout": 0.05, 
        "desc": "Balanced logic and creativity.", 
        "color": "#ffcc00"
    },
    "MAX": {
        "r": 64, "alpha": 128, "dropout": 0.1, 
        "desc": "Deep reasoning, high VRAM usage.", 
        "color": "#ff0055"
    }
}

# --- 1. MANAGEMENT FUNCTIONS ---

def list_models():
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Directory '{MODELS_DIR}' not found.")
        return
    
    print("\n🐔 --- HEN MODEL COOP ---")
    for item in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(path):
            desc_path = os.path.join(path, "hen_desc.json")
            tier = "UNKNOWN"
            status = "???"
            if os.path.exists(desc_path):
                with open(desc_path, 'r') as f:
                    data = json.load(f)
                    tier = data.get("tier", "CUSTOM")
                    status = data.get("status", "unknown")
            
            print(f"📦 {item.ljust(20)} | Tier: {tier.ljust(5)} | Status: {status}")
    print("-------------------------")

def create_model_slot(name, tier):
    tier = tier.upper()
    path = os.path.join(MODELS_DIR, name)
    
    if os.path.exists(path):
        print(f"⚠️  Model '{name}' already exists!")
        return

    os.makedirs(path, exist_ok=True)
    
    config = ARCHETYPES.get(tier, ARCHETYPES["MINI"])
    
    metadata = {
        "name": name,
        "tier": tier,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "initialized_untrained",
        "training_config": config
    }
    
    with open(os.path.join(path, "hen_desc.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Created empty shell for: {name} ({tier})")
    print(f"👉 To train it: python hen_trainer.py train {name} --dataset your_data.json")

def delete_model(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        print(f"❌ Model '{name}' does not exist.")
        return
    
    confirm = input(f"🔥 Are you sure you want to DELETE '{name}'? (y/n): ")
    if confirm.lower() == 'y':
        shutil.rmtree(path)
        print(f"🗑️  Deleted {name}")
    else:
        print("Cancelled.")

# --- 2. TRAINING ENGINE ---

def train_model(model_name, dataset_path, epochs=1):
    model_path = os.path.join(MODELS_DIR, model_name)
    desc_path = os.path.join(model_path, "hen_desc.json")
    
    # Load config or use defaults
    if os.path.exists(desc_path):
        with open(desc_path, 'r') as f:
            meta = json.load(f)
            tier_config = meta.get("training_config", ARCHETYPES["MINI"])
    else:
        print(f"⚠️  No metadata found for {model_name}. Using MINI settings.")
        tier_config = ARCHETYPES["MINI"]

    print(f"🚀 Starting training for {model_name}...")
    print(f"📊 Config: Rank {tier_config['r']} | Alpha {tier_config['alpha']}")

    # 1. Load Base Model (Quantized)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Setup LoRA
    peft_config = LoraConfig(
        r=tier_config['r'],
        lora_alpha=tier_config['alpha'],
        lora_dropout=tier_config['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Target attention layers
    )
    model = get_peft_model(model, peft_config)

    # 3. Load Dataset (Simple JSON support)
    if dataset_path.endswith(".json"):
        data = load_dataset("json", data_files=dataset_path, split="train")
    else:
        # Fallback to a standard HF dataset if not a local file
        data = load_dataset(dataset_path, split="train")

    # 4. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        dataset_text_field="text", # Ensure your JSON has a "text" field!
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=50 if epochs == 1 else -1, # Quick test or full run
            num_train_epochs=epochs,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir=f"outputs/{model_name}",
            optim="paged_adamw_8bit"
        ),
    )

    # 5. Train and Save
    print("🔥 Training started...")
    trainer.train()
    
    print(f"💾 Saving adapter to {model_path}...")
    trainer.model.save_pretrained(model_path)
    
    # Update status
    if os.path.exists(desc_path):
        with open(desc_path, 'r+') as f:
            meta = json.load(f)
            meta["status"] = "ready"
            meta["trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.seek(0)
            json.dump(meta, f, indent=4)
            f.truncate()

    print("🎉 Training Complete! You can now load this model in the Web UI.")

# --- 3. MAIN MENU ---

def main():
    parser = argparse.ArgumentParser(description="Hen-2B: Trainer & Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # gen: Create a new model slot
    gen_parser = subparsers.add_parser("gen", help="Create a new model slot")
    gen_parser.add_argument("name", help="Name of the model (e.g., hen_lite_v1)")
    gen_parser.add_argument("tier", choices=["LITE", "MINI", "MAX"], help="Model archetype")

    # train: Train a model
    train_parser = subparsers.add_parser("train", help="Train an existing model slot")
    train_parser.add_argument("name", help="Name of the model to train")
    train_parser.add_argument("--dataset", required=True, help="Path to .json dataset")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")

    # list: Show models
    subparsers.add_parser("list", help="List all models")

    # delete: Remove a model
    del_parser = subparsers.add_parser("del", help="Delete a model")
    del_parser.add_argument("name", help="Name of the model to delete")

    args = parser.parse_args()

    if args.command == "gen":
        create_model_slot(args.name, args.tier)
    elif args.command == "list":
        list_models()
    elif args.command == "del":
        delete_model(args.name)
    elif args.command == "train":
        train_model(args.name, args.dataset, args.epochs)

if __name__ == "__main__":
    main()