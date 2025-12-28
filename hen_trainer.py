import os
import json
import torch
from datetime import datetime
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainerCallback,
    logging as hf_logging
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, interleave_datasets
from trl import SFTConfig, SFTTrainer

# --- SYSTEM SETTINGS ---
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BASE_OUTPUT_DIR = "./HenModels"

# --- CUSTOM PROGRESS BAR ---
class HenProgressBar(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        print("\n" + "─" * 70)
        self.pbar = tqdm(total=state.max_steps, desc="🚀 Training Hen", unit="step", colour="yellow")

    def on_step_end(self, args, state, control, **kwargs):
        self.pbar.update(1)
        if len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                self.pbar.set_postfix({"loss": f"{last_log['loss']:.4f}"})

    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()
        print("─" * 70 + "\n")

def draw_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("╔" + "═" * 68 + "╗")
    print(f"║ {'🐔 HEN TRAINER - MODEL FORGE 🐔'.center(66)} ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║ {'Create and train new Hen models'.center(66)} ║")
    print("╚" + "═" * 68 + "╝")

def generate_model_folder_name(tier):
    """Generate unique folder name with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tier == "mini":
        return f"hen_o2_mini_{timestamp}"
    elif tier == "max":
        return f"hen_o2_max_{timestamp}"
    elif tier == "ultra":
        return f"hen_o2_ultra_{timestamp}"

def create_metadata(output_path, tier, steps, datasets_used, train_time):
    """Create hen_desc.json metadata file"""
    metadata = {
        "model_name": os.path.basename(output_path),
        "tier": tier,
        "base_model": MODEL_ID,
        "created_at": datetime.now().isoformat(),
        "training_steps": steps,
        "datasets": datasets_used,
        "training_time_minutes": round(train_time, 2),
        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "dropout": 0.1
        }
    }
    
    with open(os.path.join(output_path, "hen_desc.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📝 Metadata saved to {os.path.join(output_path, 'hen_desc.json')}")

def load_datasets(tier, tokenizer):
    """Load and format datasets based on tier"""
    datasets_info = {
        "mini": {
            "alpaca": ("train[:500]", 500),
            "oasst": ("train[:800]", 800),
            "dolly": ("train[:200]", 200)
        },
        "max": {
            "alpaca": ("train[:3000]", 3000),
            "oasst": ("train[:5000]", 5000),
            "dolly": ("train[:1500]", 1500)
        },
        "ultra": {
            "alpaca": ("train[:5000]", 5000),
            "oasst": ("train[:8000]", 8000),
            "dolly": ("train[:3000]", 3000),
            "wizardlm": ("train[:2000]", 2000),
            "ultrachat": ("train[:3000]", 3000)
        }
    }
    
    config = datasets_info[tier]
    print(f"📊 Loading datasets for {tier.upper()} tier...")
    
    # 1. Alpaca
    print("  → Alpaca...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split=config["alpaca"][0])
    alpaca = alpaca.filter(lambda x: len(x['instruction']) < 200 and len(x['output']) < 300 and len(x['output']) > 20)
    alpaca = alpaca.shuffle(seed=42).select(range(min(len(alpaca), config["alpaca"][1])))
    
    def format_alpaca(ex):
        messages = [
            {"role": "system", "content": "You are Hen, a helpful AI assistant."},
            {"role": "user", "content": ex['instruction']},
            {"role": "assistant", "content": ex['output']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    
    # 2. OpenAssistant
    print("  → OpenAssistant...")
    oasst = load_dataset("OpenAssistant/oasst1", split=config["oasst"][0])
    oasst = oasst.filter(lambda x: x['role'] == 'assistant' and 20 < len(x['text']) < 400)
    
    def format_oasst(ex):
        messages = [
            {"role": "system", "content": "You are Hen, a helpful AI assistant."},
            {"role": "user", "content": ex.get('text', 'Hello')[:200]},
            {"role": "assistant", "content": ex['text']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    
    # 3. Dolly
    print("  → Dolly...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split=config["dolly"][0])
    dolly = dolly.filter(lambda x: len(x['instruction']) < 200 and 20 < len(x['response']) < 300)
    
    def format_dolly(ex):
        messages = [
            {"role": "system", "content": "You are Hen, a helpful AI assistant."},
            {"role": "user", "content": ex['instruction']},
            {"role": "assistant", "content": ex['response']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    
    # Format datasets
    alpaca_f = alpaca.map(format_alpaca, remove_columns=alpaca.column_names)
    oasst_f = oasst.map(format_oasst, remove_columns=oasst.column_names)
    dolly_f = dolly.map(format_dolly, remove_columns=dolly.column_names)
    
    datasets_list = [alpaca_f, oasst_f, dolly_f]
    probabilities = [0.3, 0.5, 0.2]
    dataset_names = ["alpaca", "oasst", "dolly"]
    
    # Ultra tier: Add WizardLM and UltraChat
    if tier == "ultra":
        print("  → WizardLM...")
        try:
            wizard = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split=config["wizardlm"][0])
            wizard = wizard.filter(lambda x: len(x.get('instruction', '')) < 200 and 20 < len(x.get('output', '')) < 300)
            
            def format_wizard(ex):
                messages = [
                    {"role": "system", "content": "You are Hen, a helpful AI assistant."},
                    {"role": "user", "content": ex.get('instruction', '')},
                    {"role": "assistant", "content": ex.get('output', '')}
                ]
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
            
            wizard_f = wizard.map(format_wizard, remove_columns=wizard.column_names)
            datasets_list.append(wizard_f)
            dataset_names.append("wizardlm")
        except Exception as e:
            print(f"    ⚠️ WizardLM failed: {e}")
        
        print("  → UltraChat...")
        try:
            ultra = load_dataset("stingning/ultrachat", split=config["ultrachat"][0])
            ultra = ultra.filter(lambda x: len(str(x.get('data', [''])[0])) < 200)
            
            def format_ultra(ex):
                messages = [
                    {"role": "system", "content": "You are Hen, a helpful AI assistant."},
                    {"role": "user", "content": str(ex.get('data', [''])[0])[:200]},
                    {"role": "assistant", "content": str(ex.get('data', ['', ''])[1])[:300]}
                ]
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
            
            ultra_f = ultra.map(format_ultra, remove_columns=ultra.column_names)
            datasets_list.append(ultra_f)
            dataset_names.append("ultrachat")
            probabilities = [0.2, 0.35, 0.15, 0.15, 0.15]
        except Exception as e:
            print(f"    ⚠️ UltraChat failed: {e}")
    
    # Interleave datasets
    print("  → Merging datasets...")
    combined = interleave_datasets(datasets_list, probabilities=probabilities, seed=42)
    
    # Split for evaluation
    split_data = combined.train_test_split(test_size=0.1, seed=42)
    
    print(f"✅ Train: {len(split_data['train'])} | Eval: {len(split_data['test'])}")
    return split_data['train'], split_data['test'], dataset_names

def train_model(tier):
    """Main training function"""
    draw_banner()
    
    # Create unique output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(BASE_OUTPUT_DIR, generate_model_folder_name(tier))
    os.makedirs(output_path, exist_ok=True)
    
    print(f"📁 Output directory: {output_path}\n")
    
    # Training configs by tier
    tier_configs = {
        "mini": {"steps": 20, "lr": 3e-4},
        "max": {"steps": 200, "lr": 2e-4},
        "ultra": {"steps": 400, "lr": 1.5e-4}
    }
    
    config = tier_configs[tier]
    
    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("🤖 Loading Qwen2.5-3B (4-Bit)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    print("🛠️ Applying LoRA...")
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1, 
        task_type="CAUSAL_LM", 
        bias="none"
    )
    model = get_peft_model(model, peft_config)

    # Load datasets
    train_ds, eval_ds, dataset_names = load_datasets(tier, tokenizer)

    # Training config
    sft_config = SFTConfig(
        output_dir=output_path,
        max_steps=config["steps"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=config["lr"],
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(config["steps"] // 5, 10),
        save_strategy="steps",
        save_steps=max(config["steps"] // 5, 10),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        disable_tqdm=True,
        dataset_text_field="text",
        max_seq_length=256,
        report_to="none",
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_total_limit=2
    )

    trainer = SFTTrainer(
        model=model, 
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[HenProgressBar()]
    )
    
    print(f"\n🔥 TRAINING {tier.upper()} MODEL...")
    print(f"📈 Steps: {config['steps']} | LR: {config['lr']}\n")
    
    import time
    start_time = time.time()
    trainer.train()
    train_time = (time.time() - start_time) / 60
    
    # Save final model
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Create metadata
    create_metadata(output_path, tier, config["steps"], dataset_names, train_time)
    
    print(f"\n✅ Model saved to: {output_path}")
    print(f"⏱️  Training time: {train_time:.2f} minutes")
    input("\nPress Enter to continue...")

def main():
    while True:
        draw_banner()
        print("\n 🐣 HEN MODEL TIERS:\n")
        print(" 1. o2 MINI   - Quick test model (~1-2 min, 20 steps)")
        print(" 2. o2 MAX    - Production model (~7-10 min, 200 steps)")
        print(" 3. o2 ULTRA  - Maximum quality (~15-20 min, 400 steps, 5 datasets)")
        print(" 4. Exit\n")
        
        choice = input("Select tier to train: ")
        
        if choice == '1':
            train_model("mini")
        elif choice == '2':
            train_model("max")
        elif choice == '3':
            train_model("ultra")
        elif choice == '4':
            break

if __name__ == "__main__":
    main()