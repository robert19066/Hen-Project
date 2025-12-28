import os
import json
import torch
import sys
import textwrap
from googletrans import Translator
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    logging as hf_logging
)
from peft import PeftModel

# --- SYSTEM SETTINGS ---
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_DIR = "./HenModels"

session = PromptSession()
bindings = KeyBindings()

MULTILINE_MODE = False
ROMANIAN_MODE = False
CURRENT_TEMP = 0.7
SELECTED_MODEL = None

# --- KEY BINDINGS ---
@bindings.add('c-o')
def _(event):
    global MULTILINE_MODE
    MULTILINE_MODE = not MULTILINE_MODE
    print(f"\n🔄 MODE: {'MULTI' if MULTILINE_MODE else 'SINGLE'}")

@bindings.add('c-r')
def _(event):
    global ROMANIAN_MODE
    ROMANIAN_MODE = not ROMANIAN_MODE
    print(f"\n🇷🇴 ROMANIAN: {'ON' if ROMANIAN_MODE else 'OFF'}")

# --- UI FUNCTIONS ---
def draw_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("╔" + "═" * 68 + "╗")
    print(f"║ {'🐔 HEN CHAT - MODEL SELECTOR 🐔'.center(66)} ║")
    print("╠" + "═" * 68 + "╣")
    if SELECTED_MODEL:
        model_name = os.path.basename(SELECTED_MODEL)
        print(f"║ {'Current: ' + model_name[:56]:<66} ║")
    lang = "Romanian" if ROMANIAN_MODE else "English"
    status = f"Mode: {'Multi' if MULTILINE_MODE else 'Single'} | Lang: {lang} | Temp: {CURRENT_TEMP}"
    print(f"║ {status.center(66)} ║")
    print("╚" + "═" * 68 + "╝")

def get_user_input():
    print(f"\n[Mode: {'MULTI' if MULTILINE_MODE else 'SINGLE'} | Lang: {'RO' if ROMANIAN_MODE else 'EN'}]")
    print("(Ctrl+O: Mode | Ctrl+R: Romanian | Ctrl+C: Menu)")

    if MULTILINE_MODE:
        print("[--- MULTI-LINE: Type 'END' to finish ---]")
        lines = []
        while True:
            line = session.prompt('→ ', key_bindings=bindings)
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        return "\n".join(lines)
    else:
        return session.prompt('→ ', key_bindings=bindings)

def discover_models():
    """Scan for trained models and return list with metadata"""
    if not os.path.exists(BASE_MODEL_DIR):
        return []
    
    models = []
    for folder in os.listdir(BASE_MODEL_DIR):
        folder_path = os.path.join(BASE_MODEL_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Check if it's a valid model (has adapter_config.json)
        if not os.path.exists(os.path.join(folder_path, "adapter_config.json")):
            continue
        
        # Try to load metadata
        metadata_path = os.path.join(folder_path, "hen_desc.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                models.append({
                    "path": folder_path,
                    "name": folder,
                    "metadata": metadata
                })
            except:
                # If metadata fails, add basic info
                models.append({
                    "path": folder_path,
                    "name": folder,
                    "metadata": {"tier": "unknown", "created_at": "unknown"}
                })
        else:
            models.append({
                "path": folder_path,
                "name": folder,
                "metadata": {"tier": "unknown", "created_at": "unknown"}
            })
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x['metadata'].get('created_at', ''), reverse=True)
    return models

def select_model():
    """Display model selection menu"""
    global SELECTED_MODEL
    
    draw_banner()
    models = discover_models()
    
    if not models:
        print("\n❌ No trained models found!")
        print(f"📁 Models should be in: {BASE_MODEL_DIR}")
        print("\nRun hen_trainer.py first to create models.\n")
        input("Press Enter to continue...")
        return False
    
    print(f"\n📦 Found {len(models)} trained model(s):\n")
    
    for idx, model in enumerate(models, 1):
        meta = model['metadata']
        tier = meta.get('tier', 'unknown').upper()
        created = meta.get('created_at', 'unknown')
        if created != 'unknown':
            created = created.split('T')[0]  # Just the date
        
        steps = meta.get('training_steps', '?')
        datasets = ', '.join(meta.get('datasets', ['?']))
        
        print(f" {idx}. {model['name']}")
        print(f"    Tier: {tier} | Steps: {steps} | Created: {created}")
        print(f"    Datasets: {datasets}\n")
    
    print(" 0. Use Base Model (No fine-tuning)")
    print(" B. Back to main menu\n")
    
    choice = input("Select model number: ").strip()
    
    if choice.upper() == 'B':
        return False
    elif choice == '0':
        SELECTED_MODEL = None
        print("\n✅ Will use base Qwen 2.5 model (no fine-tuning)")
        input("Press Enter to continue...")
        return True
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                SELECTED_MODEL = models[idx]['path']
                print(f"\n✅ Selected: {models[idx]['name']}")
                input("Press Enter to start chatting...")
                return True
            else:
                print("\n❌ Invalid selection!")
                input("Press Enter to try again...")
                return False
        except ValueError:
            print("\n❌ Invalid input!")
            input("Press Enter to try again...")
            return False

def run_chat():
    """Main chat loop"""
    global ROMANIAN_MODE, CURRENT_TEMP
    
    draw_banner()
    
    print("\n🤖 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    if SELECTED_MODEL:
        print(f"📦 Loading fine-tuned model from {os.path.basename(SELECTED_MODEL)}...")
        model = PeftModel.from_pretrained(base_model, SELECTED_MODEL)
        print("✅ Fine-tuned Hen loaded!")
    else:
        model = base_model
        print("✅ Base Qwen 2.5 loaded (no fine-tuning)")

    translator = Translator()
    conversation_history = []

    print("\n💬 Chat started! Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_msg = get_user_input()
        except KeyboardInterrupt:
            print("\n\n👋 Chat ended by user.")
            break

        if user_msg.lower() in ["exit", "quit"]:
            print("\n👋 Goodbye!")
            break
        
        if user_msg.strip() == "":
            continue

        original_user_msg = user_msg

        # Translate input if Romanian mode
        if ROMANIAN_MODE:
            try:
                user_msg = translator.translate(user_msg, src="ro", dest="en").text
            except Exception:
                pass

        # System prompt
        sys_msg = (
            "You are Hen, a helpful AI assistant.\n"
            "Rules:\n"
            "No markdown.\n"
            "No bullet points.\n"
            "No headings.\n"
            "Plain text only.\n"
            "If code is requested, output raw code only.\n"
        )

        messages = [{"role": "system", "content": sys_msg}]
        messages.extend(conversation_history[-2:])  # Last 1 exchange
        messages.append({"role": "user", "content": user_msg})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=CURRENT_TEMP,
                do_sample=True if CURRENT_TEMP > 0 else False,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "assistant" in raw_output.lower():
            response = raw_output.split("assistant")[-1].strip()
        else:
            response = raw_output.strip()

        # Clean markdown artifacts
        for token in ["```", "**", "###", "* ", "- ", "`", "<|im_end|>", "<|im_start|>"]:
            response = response.replace(token, "")

        # Translate output if Romanian mode
        if ROMANIAN_MODE:
            try:
                response = translator.translate(response, src="en", dest="ro").text
            except Exception:
                response = "[Translation error]\n" + response

        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": response})

        # Display response
        print("┌─[ HEN ]" + "─" * 55 + "┐")
        for line in textwrap.wrap(response, width=63):
            print(f"│ {line.ljust(63)} │")
        print("└" + "─" * 67 + "┘")

    input("\nPress Enter to return to menu...")

def change_settings():
    """Settings menu"""
    global CURRENT_TEMP, ROMANIAN_MODE, MULTILINE_MODE
    
    draw_banner()
    print("\n⚙️  SETTINGS:\n")
    print(f" 1. Temperature: {CURRENT_TEMP}")
    print(f" 2. Romanian Mode: {'ON' if ROMANIAN_MODE else 'OFF'}")
    print(f" 3. Input Mode: {'MULTI-LINE' if MULTILINE_MODE else 'SINGLE-LINE'}")
    print(" 4. Back\n")
    
    choice = input("Select setting to change: ")
    
    if choice == '1':
        try:
            new_temp = float(input("Enter temperature (0.0-2.0): "))
            if 0 <= new_temp <= 2.0:
                CURRENT_TEMP = new_temp
                print(f"✅ Temperature set to {CURRENT_TEMP}")
            else:
                print("❌ Temperature must be between 0.0 and 2.0")
        except ValueError:
            print("❌ Invalid number")
        input("\nPress Enter...")
    elif choice == '2':
        ROMANIAN_MODE = not ROMANIAN_MODE
        print(f"✅ Romanian mode: {'ON' if ROMANIAN_MODE else 'OFF'}")
        input("\nPress Enter...")
    elif choice == '3':
        MULTILINE_MODE = not MULTILINE_MODE
        print(f"✅ Input mode: {'MULTI-LINE' if MULTILINE_MODE else 'SINGLE-LINE'}")
        input("\nPress Enter...")

def main():
    while True:
        draw_banner()
        print("\n 🐔 HEN CHAT MENU:\n")
        print(" 1. Select Model")
        print(" 2. Start Chat")
        print(" 3. Settings")
        print(" 4. Exit\n")
        
        choice = input("Selection: ")
        
        if choice == '1':
            select_model()
        elif choice == '2':
            if SELECTED_MODEL is None:
                print("\n⚠️  No model selected! Select a model first or use base model.")
                input("Press Enter...")
            else:
                run_chat()
        elif choice == '3':
            change_settings()
        elif choice == '4':
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()