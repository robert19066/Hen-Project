import os
import json
import torch
import time
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

##########################################################################################################################
# HEN TRAINER V3 - Enhanced Edition                                                                                      #
# Major Updates:                                                                                                          #
# - Better datasets (Code-Feedback, Python 18k, etc.)                                                                    #
# - QwQ-32B reasoning base model support                                                                                 #
# - More model tiers (Lite, Pro, Vision prep)                                                                            #                                                                         #
##########################################################################################################################

hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model configurations by tier
MODEL_CONFIGS = {
    "standard": "Qwen/Qwen2.5-3B-Instruct",      # Chat models
    "reasoning": "Qwen/QwQ-32B-Preview",          # Reasoning models (VRAM warning!)
    "lite": "Qwen/Qwen2.5-1.5B-Instruct",        # For low VRAM
}

BASE_OUTPUT_DIR = "/HenModels"

class HenProgressBar(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "═" * 70)
        self.pbar = tqdm(total=state.max_steps, desc="🐔 Training Hen", unit="step", colour="yellow")
        
    def on_step_end(self, args, state, control, **kwargs):
        self.pbar.update(1)
        if len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                elapsed = time.time() - self.start_time
                steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
                self.pbar.set_postfix({
                    "loss": f"{last_log['loss']:.4f}",
                    "step/s": f"{steps_per_sec:.2f}"
                })
                
    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time/60:.2f} minutes")
        print("═" * 70 + "\n")

def draw_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("╔" + "═" * 68 + "╗")
    print(f"║ {'HEN TRAINER V3 - ENHANCED'.center(66)} ║")
    print("╠" + "═" * 68 + "╣")
    print("║ " + "RTX 4050 Optimized | Better Datasets | More Models".center(66) + " ║")
    print("╚" + "═" * 68 + "╝")

def generate_model_folder_name(tier):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tier_names = {
        # LITE TIER (Low VRAM)
        "lite": f"hen_lite_{timestamp}",
        "lite_code": f"hen_lite_code_{timestamp}",
        
        # STANDARD TIER (3B base)
        "mini": f"hen_o2_mini_{timestamp}",
        "max": f"hen_o2_max_{timestamp}",
        "ultra": f"hen_o2_ultra_{timestamp}",
        "pro": f"hen_pro_balanced_{timestamp}",
        
        # CODE TIER (Enhanced datasets)
        "code": f"hen_code_{timestamp}",
        "code_max": f"hen_code_max_{timestamp}",
        "code_expert": f"hen_code_expert_{timestamp}",
        
        # REASONING TIER (QwQ base - HIGH VRAM!)
        "r1_mini": f"hen_r1_mini_{timestamp}",
        "r1_max": f"hen_r1_max_{timestamp}",
        "r2_deep": f"hen_r2_deep_{timestamp}",
        
        # SPECIALIZED
        "oni": f"hen_oni_universal_{timestamp}",
        "math": f"hen_math_specialist_{timestamp}",
    }
    return tier_names.get(tier, f"hen_custom_{timestamp}")

def create_metadata(output_path, tier, steps, datasets_used, train_time, base_model):
    is_reasoning = tier in ["r1_mini", "r1_max", "r2_deep"]
    is_code = "code" in tier
    is_lite = "lite" in tier
    
    metadata = {
        "model_name": os.path.basename(output_path),
        "tier": tier,
        "base_model": base_model,
        "created_at": datetime.now().isoformat(),
        "training_steps": steps,
        "datasets": datasets_used,
        "training_time_minutes": round(train_time, 2),
        "capabilities": {
            "reasoning": is_reasoning,
            "code_specialized": is_code,
            "universal": tier == "oni",
            "lightweight": is_lite,
            "math_focused": tier == "math"
        },
        "hardware_requirements": {
            "min_vram_gb": 6 if is_lite else (16 if is_reasoning else 8),
            "quantization": "4bit",
            "recommended_gpu": "RTX 4050+" if not is_reasoning else "RTX 4090 / A100"
        }
    }
    
    with open(os.path.join(output_path, "hen_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to hen_metadata.json")

def load_datasets(tier, tokenizer):
    """Enhanced dataset loading with better sources"""
    
    # ============== FORMATTERS ==============
    def standard_fmt(ex):
        """Standard chat formatter"""
        instruction = ex.get('instruction', ex.get('prompt', ex.get('text', '')))
        output = ex.get('output', ex.get('response', ex.get('completion', '')))
        
        messages = [
            {"role": "system", "content": "You are Hen, a helpful AI assistant."},
            {"role": "user", "content": str(instruction)[:600]},
            {"role": "assistant", "content": str(output)[:800]}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    
    def code_fmt(ex):
        """Code-specific formatter"""
        instruction = ex.get('instruction', ex.get('prompt', ''))
        output = ex.get('output', ex.get('completion', ''))
        
        messages = [
            {"role": "system", "content": "You are Hen Code, an expert programming assistant. Provide clear, efficient, and well-documented code."},
            {"role": "user", "content": str(instruction)},
            {"role": "assistant", "content": str(output)}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    
    def reasoning_fmt(ex):
        """Deep reasoning formatter with <think> tags"""
        question = ex.get('question', ex.get('instruction', ex.get('problem', '')))
        answer = ex.get('answer', ex.get('solution', ''))
        
        # Extract reasoning if present
        if '####' in str(answer):
            reasoning = answer.split('####')[0].strip()
            final_answer = answer.split('####')[-1].strip()
        else:
            reasoning = answer
            final_answer = answer
        
        messages = [
            {"role": "system", "content": "You are Hen R2, a reasoning AI. Always think step-by-step inside <think> tags before providing your final answer."},
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\nFinal Answer: {final_answer}"}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    
    # ============== DATASET CONFIGURATIONS ==============
    print(f"📦 Loading datasets for {tier.upper()}...")
    datasets_list = []
    probabilities = []
    dataset_names = []
    
    try:
        # ===== LITE MODELS (Low VRAM) =====
        if tier == "lite":
            print("  → Lite balanced mix (1.5B base)")
            d1 = load_dataset("yahma/alpaca-cleaned", split="train[:1000]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("OpenAssistant/oasst1", split="train[:800]").filter(lambda x: x.get('role') == 'assistant').map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2]
            probabilities = [0.6, 0.4]
            dataset_names = ["alpaca_lite", "oasst_lite"]
        
        elif tier == "lite_code":
            print("  → Lite code focus")
            d1 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:1500]").map(code_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("yahma/alpaca-cleaned", split="train[:300]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2]
            probabilities = [0.85, 0.15]
            dataset_names = ["python_18k", "alpaca_base"]
        
        # ===== STANDARD MODELS (3B Qwen) =====
        elif tier == "mini":
            print("  → Mini quick training")
            d1 = load_dataset("yahma/alpaca-cleaned", split="train[:800]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("OpenAssistant/oasst1", split="train[:1000]").filter(lambda x: x.get('role') == 'assistant').map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2]
            probabilities = [0.5, 0.5]
            dataset_names = ["alpaca", "oasst1"]
        
        elif tier == "max":
            print("  → Max production quality")
            d1 = load_dataset("yahma/alpaca-cleaned", split="train[:3000]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("OpenAssistant/oasst1", split="train[:5000]").filter(lambda x: x.get('role') == 'assistant').map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d3 = load_dataset("databricks/databricks-dolly-15k", split="train[:2000]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2, d3]
            probabilities = [0.4, 0.4, 0.2]
            dataset_names = ["alpaca", "oasst1", "dolly"]
        
        elif tier == "ultra":
            print("  → Ultra high quality")
            d1 = load_dataset("yahma/alpaca-cleaned", split="train[:5000]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("OpenAssistant/oasst1", split="train[:8000]").filter(lambda x: x.get('role') == 'assistant').map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d3 = load_dataset("databricks/databricks-dolly-15k", split="train[:3000]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2, d3]
            probabilities = [0.35, 0.45, 0.2]
            dataset_names = ["alpaca", "oasst1", "dolly"]
        
        elif tier == "pro":
            print("  → Pro balanced (chat + light code + logic)")
            d1 = load_dataset("yahma/alpaca-cleaned", split="train[:3000]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]").map(code_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d3 = load_dataset("gsm8k", "main", split="train[:1500]").map(reasoning_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2, d3]
            probabilities = [0.5, 0.3, 0.2]
            dataset_names = ["alpaca", "python_code", "gsm8k"]
        
        # ===== CODE MODELS (Enhanced datasets) =====
        elif tier == "code":
            print("  → Code model (Python 18k + Code 122k)")
            d1 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:5000]").map(code_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train[:4000]").map(code_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d3 = load_dataset("yahma/alpaca-cleaned", split="train[:800]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2, d3]
            probabilities = [0.5, 0.4, 0.1]
            dataset_names = ["python_18k", "code_122k", "alpaca_speech"]
        
        elif tier == "code_max":
            print("  → Code Max (Heavy training)")
            d1 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:8000]").map(code_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train[:7000]").map(code_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d3 = load_dataset("yahma/alpaca-cleaned", split="train[:1500]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2, d3]
            probabilities = [0.45, 0.45, 0.1]
            dataset_names = ["python_18k", "code_122k", "alpaca_speech"]
        
        elif tier == "code_expert":
            print("  → Code Expert (90% pure code)")
            d1 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:10000]").map(code_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train[:10000]").map(code_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d3 = load_dataset("yahma/alpaca-cleaned", split="train[:500]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2, d3]
            probabilities = [0.5, 0.4, 0.1]
            dataset_names = ["python_18k_full", "code_122k_full", "alpaca_minimal"]
        
        # ===== REASONING MODELS (QwQ-32B base - WARNING: HIGH VRAM!) =====
        elif tier == "r1_mini":
            print("  ⚠️  Reasoning Mini (QwQ-32B - needs 16GB+ VRAM!)")
            d1 = load_dataset("gsm8k", "main", split="train[:2000]").map(reasoning_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("yahma/alpaca-cleaned", split="train[:500]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2]
            probabilities = [0.85, 0.15]
            dataset_names = ["gsm8k_reasoning", "alpaca_base"]
        
        elif tier == "r1_max":
            print("  ⚠️  Reasoning Max (QwQ-32B - needs 20GB+ VRAM!)")
            d1 = load_dataset("gsm8k", "main", split="train[:5000]").map(reasoning_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("yahma/alpaca-cleaned", split="train[:1000]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2]
            probabilities = [0.9, 0.1]
            dataset_names = ["gsm8k_heavy", "alpaca_base"]
        
        elif tier == "r2_deep":
            print("  ⚠️  R2 Deep Reasoning (QwQ-32B - needs 24GB+ VRAM!)")
            d1 = load_dataset("gsm8k", "main", split="train[:6000]").map(reasoning_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            try:
                d2 = load_dataset("hendrycks/competition_math", split="train[:2000]").map(reasoning_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
                datasets_list = [d1, d2]
                probabilities = [0.6, 0.4]
                dataset_names = ["gsm8k_deep", "competition_math"]
            except:
                print("  ⚠️  MATH dataset failed, using more GSM8K")
                d2 = load_dataset("gsm8k", "main", split="train[6000:10000]").map(reasoning_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
                datasets_list = [d1, d2]
                probabilities = [0.7, 0.3]
                dataset_names = ["gsm8k_part1", "gsm8k_part2"]
        
        # ===== SPECIALIZED MODELS =====
        elif tier == "oni":
            print("  → Oni Universal (Code + Math + Chat)")
            d1 = load_dataset("yahma/alpaca-cleaned", split="train[:2500]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2500]").map(code_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d3 = load_dataset("gsm8k", "main", split="train[:2000]").map(reasoning_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2, d3]
            probabilities = [0.4, 0.35, 0.25]
            dataset_names = ["alpaca", "python_code", "gsm8k"]
        
        elif tier == "math":
            print("  → Math Specialist")
            d1 = load_dataset("gsm8k", "main", split="train[:5000]").map(reasoning_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            d2 = load_dataset("yahma/alpaca-cleaned", split="train[:1000]").map(standard_fmt, remove_columns=lambda x: [c for c in x if c != 'text'])
            datasets_list = [d1, d2]
            probabilities = [0.9, 0.1]
            dataset_names = ["gsm8k_math", "alpaca_base"]
        
        else:
            raise ValueError(f"Unknown tier: {tier}")
        
        # Clean up column names
        for i, ds in enumerate(datasets_list):
            cols_to_remove = [c for c in ds.column_names if c != 'text']
            if cols_to_remove:
                datasets_list[i] = ds.remove_columns(cols_to_remove)
        
        # Merge datasets
        print("  → Merging and splitting datasets...")
        combined = interleave_datasets(datasets_list, probabilities=probabilities, seed=42)
        split_data = combined.train_test_split(test_size=0.05, seed=42)
        
        print(f"  ✓ Loaded {len(combined)} total examples")
        return split_data['train'], split_data['test'], dataset_names
        
    except Exception as e:
        print(f"  ✗ Error loading datasets: {e}")
        raise

def check_vram():
    """Check available VRAM"""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {vram_gb:.1f}GB")
        return vram_gb
    else:
        print("⚠️  No CUDA GPU detected!")
        return 0

def train_model(tier):
    draw_banner()
    
    # Check VRAM first
    vram = check_vram()
    
    # Determine base model
    if tier in ["r1_mini", "r1_max", "r2_deep"]:
        base_model = MODEL_CONFIGS["reasoning"]
        if vram < 16:
            print(f"\n⚠️  WARNING: Reasoning models need 16GB+ VRAM, you have {vram:.1f}GB")
            print("This will likely crash. Use 'lite' or standard models instead.")
            choice = input("Continue anyway? (y/n): ")
            if choice.lower() != 'y':
                return
    elif tier in ["lite", "lite_code"]:
        base_model = MODEL_CONFIGS["lite"]
    else:
        base_model = MODEL_CONFIGS["standard"]
    
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(BASE_OUTPUT_DIR, generate_model_folder_name(tier))
    os.makedirs(output_path, exist_ok=True)
    print(f"📁 Output: {output_path}\n")

    # Training configs optimized for RTX 4050
    tier_configs = {
        # Lite models (1.5B)
        "lite": {"steps": 50, "lr": 3e-4, "batch": 4, "grad_accum": 2},
        "lite_code": {"steps": 80, "lr": 2e-4, "batch": 4, "grad_accum": 2},
        
        # Standard models (3B)
        "mini": {"steps": 50, "lr": 3e-4, "batch": 2, "grad_accum": 4},
        "max": {"steps": 250, "lr": 2e-4, "batch": 2, "grad_accum": 4},
        "ultra": {"steps": 500, "lr": 1.5e-4, "batch": 2, "grad_accum": 4},
        "pro": {"steps": 300, "lr": 2e-4, "batch": 2, "grad_accum": 4},
        
        # Code models
        "code": {"steps": 300, "lr": 2e-4, "batch": 2, "grad_accum": 4},
        "code_max": {"steps": 600, "lr": 1.5e-4, "batch": 2, "grad_accum": 4},
        "code_expert": {"steps": 800, "lr": 1e-4, "batch": 2, "grad_accum": 4},
        
        # Reasoning models (32B - HEAVY!)
        "r1_mini": {"steps": 100, "lr": 1e-4, "batch": 1, "grad_accum": 8},
        "r1_max": {"steps": 300, "lr": 8e-5, "batch": 1, "grad_accum": 8},
        "r2_deep": {"steps": 400, "lr": 5e-5, "batch": 1, "grad_accum": 8},
        
        # Specialized
        "oni": {"steps": 250, "lr": 2e-4, "batch": 2, "grad_accum": 4},
        "math": {"steps": 350, "lr": 1.5e-4, "batch": 2, "grad_accum": 4},
    }
    
    config = tier_configs[tier]
    
    print(f"🔧 Loading {base_model}...")
    start_time = time.time()
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True  # Extra memory savings
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",  # Changed from {"": 0} for better memory management
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    print("🔗 Applying LoRA...")
    # Optimized LoRA config for 4050
    peft_config = LoraConfig(
        r=8,  # Reduced from 16 for speed
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # More coverage
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    train_ds, eval_ds, dataset_names = load_datasets(tier, tokenizer)
    
    # Optimized training config
    sft_config = SFTConfig(
        output_dir=output_path,
        max_steps=config["steps"],
        per_device_train_batch_size=config["batch"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"],
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(50, config["steps"] // 4),  # Save 4 checkpoints
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=max(25, config["steps"] // 8),
        dataset_text_field="text",
        max_seq_length=384,  # Reduced for speed
        packing=False,
        gradient_checkpointing=True,  # Memory savings
        optim="adamw_torch_fused",  # Faster optimizer
        lr_scheduler_type="cosine",
        warmup_steps=min(50, config["steps"] // 10),
        dataloader_num_workers=2,  # Parallel loading
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[HenProgressBar()]
    )
    
    print(f"\n🚀 Starting training ({config['steps']} steps)...\n")
    trainer.train()
    
    total_time = (time.time() - start_time) / 60
    
    print("\n💾 Saving model...")
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    create_metadata(output_path, tier, config["steps"], dataset_names, total_time, base_model)
    
    print(f"\n✅ Model saved to: {output_path}")
    print(f"⏱️  Total time: {total_time:.2f} minutes")
    input("\nPress Enter to continue...")

def main():
    while True:
        draw_banner()
        vram = check_vram()
        
        print("\n🐔 HEN MODEL TIERS:\n")
        print("═══ LITE (Low VRAM - 1.5B) ═══")
        print(" 1. Lite            (Quick, 6GB VRAM)")
        print(" 2. Lite Code       (Code focus, 6GB VRAM)")
        
        print("\n═══ STANDARD (3B Qwen) ═══")
        print(" 3. Mini            (Fast test, 8GB VRAM)")
        print(" 4. Max             (Production, 8GB VRAM)")
        print(" 5. Ultra           (High quality, 8GB VRAM)")
        print(" 6. Pro             (Balanced mix, 8GB VRAM)")
        
        print("\n═══ CODE (Enhanced Datasets) ═══")
        print(" 7. Code            (Python 18k, 8GB VRAM)")
        print(" 8. Code Max        (Heavy training, 8GB VRAM)")
        print(" 9. Code Expert     (90% pure code, 8GB VRAM)")
        
        print("\n═══ REASONING (QwQ-32B - ⚠️ HIGH VRAM!) ═══")
        if vram < 16:
            print(" 10. R1 Mini        (⚠️ NEEDS 16GB+ - YOU HAVE {:.1f}GB)".format(vram))
            print(" 11. R1 Max         (⚠️ NEEDS 20GB+ - YOU HAVE {:.1f}GB)".format(vram))
            print(" 12. R2 Deep        (⚠️ NEEDS 24GB+ - YOU HAVE {:.1f}GB)".format(vram))
        else:
            print(" 10. R1 Mini        (Reasoning, 16GB+ VRAM)")
            print(" 11. R1 Max         (Heavy reasoning, 20GB+ VRAM)")
            print(" 12. R2 Deep        (Math expert, 24GB+ VRAM)")
        
        print("\n═══ SPECIALIZED ═══")
        print(" 13. Oni Universal  (Code+Math+Chat, 8GB VRAM)")
        print(" 14. Math           (Math specialist, 8GB VRAM)")
        
        print("\n 0. Exit")
        
        c = input("\n→ Select tier: ").strip()
        
        if c == '1': train_model("lite")
        elif c == '2': train_model("lite_code")
        elif c == '3': train_model("mini")
        elif c == '4': train_model("max")
        elif c == '5': train_model("ultra")
        elif c == '6': train_model("pro")
        elif c == '7': train_model("code")
        elif c == '8': train_model("code_max")
        elif c == '9': train_model("code_expert")
        elif c == '10': train_model("r1_mini")
        elif c == '11': train_model("r1_max")
        elif c == '12': train_model("r2_deep")
        elif c == '13': train_model("oni")
        elif c == '14': train_model("math")
        elif c == '0': 
            print("\n👋 Thanks for using Hen Trainer V3!")
            break
        else:
            print("Invalid choice!")
            time.sleep(1)

if __name__ == "__main__":
    main()