import os
import json
import torch
import sys
import time
import textwrap
import threading
from googletrans import Translator
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging
)
from peft import PeftModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.theme import Theme

#############################################################################
# hen chat codeh, made by Brickboss                                         #
# if you copy this code, please give credit or ill pew pew you with my aura #
# please dont copy this code without crediting me, thank you :)             #
# Note:but if you will steal it its apache licensed so you are cooked       #
#############################################################################

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "hen": "bold yellow"
})
console = Console(theme=custom_theme)


hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_DIR = "./HenModels"

session = PromptSession()
bindings = KeyBindings()
ROMANIAN_MODE = False
CURRENT_TEMP = 0.7
SELECTED_MODEL = None


@bindings.add('c-r')
def _(event):
    global ROMANIAN_MODE
    ROMANIAN_MODE = not ROMANIAN_MODE
    state = "ON" if ROMANIAN_MODE else "OFF"
    console.print(f"\n[info]🇷🇴 Romanian Mode: {state}[/info]")


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def type_effect(text):
    """Simulate streaming text output"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.005)
    print()

def discover_models():
    if not os.path.exists(BASE_MODEL_DIR): return []
    models = []
    for folder in os.listdir(BASE_MODEL_DIR):
        folder_path = os.path.join(BASE_MODEL_DIR, folder)
        if not os.path.isdir(folder_path) or not os.path.exists(os.path.join(folder_path, "adapter_config.json")):
            continue
        
        metadata = {"tier": "unknown", "created_at": "unknown"}
        meta_path = os.path.join(folder_path, "hen_desc.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f: metadata = json.load(f)
            except: pass
            
        models.append({"path": folder_path, "name": folder, "metadata": metadata})
    models.sort(key=lambda x: x['metadata'].get('created_at', ''), reverse=True)
    return models

def select_model():
    global SELECTED_MODEL
    clear_screen()
    console.print(Panel.fit("[bold yellow]🐔 HEN CHAT - MODEL SELECTOR[/bold yellow]", border_style="yellow"))
    
    models = discover_models()
    if not models:
        console.print("[warning]❌ No fine-tuned models found in ./HenModels[/warning]")
        console.print("[dim]Run the trainer first.[/dim]")
        return

    for idx, model in enumerate(models, 1):
        meta = model['metadata']
        caps = meta.get('capabilities', {})
        icon = "🧠" if caps.get('reasoning') else "💻" if caps.get('code_specialized') else "🤖"
        
        console.print(f"[bold cyan]{idx}. {model['name']}[/bold cyan] {icon}")
        console.print(f"   [dim]Tier: {meta.get('tier')} | Created: {meta.get('created_at', '').split('T')[0]}[/dim]")
    
    console.print("\n[bold cyan]0. Use Base Model (No Fine-tuning)[/bold cyan]")
    
    choice = input("\nSelect model number: ").strip()
    if choice == '0':
        SELECTED_MODEL = None
        console.print("[green]✅ Selected Base Qwen 2.5[/green]")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                SELECTED_MODEL = models[idx]['path']
                console.print(f"[green]✅ Selected: {models[idx]['name']}[/green]")
            else:
                console.print("[red]Invalid selection[/red]")
        except:
            console.print("[red]Invalid input[/red]")
    
    time.sleep(1)

def run_chat():
    global ROMANIAN_MODE
    clear_screen()
    

    with console.status("[bold yellow]🐔 Hatching the Hen (Loading Model)...[/bold yellow]", spinner="dots12"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True
        )
        
        if SELECTED_MODEL:
            model = PeftModel.from_pretrained(base_model, SELECTED_MODEL)
            model_name = os.path.basename(SELECTED_MODEL)
        else:
            model = base_model
            model_name = "Base Qwen 2.5"

    translator = Translator()
    conversation_history = []
    
    console.print(Panel(f"[bold green]Connected to {model_name}[/bold green]\n[dim]Ctrl+R: Toggle Romanian | Type 'exit' to quit[/dim]", border_style="green"))

    while True:
        try:
            lang_indicator = "[RO]" if ROMANIAN_MODE else "[EN]"
            user_msg = session.prompt(f'\n👤 {lang_indicator} You: ', key_bindings=bindings).strip()
            
            if user_msg.lower() in ["exit", "quit"]: break
            if not user_msg: continue

            # Translate Input
            if ROMANIAN_MODE:
                with console.status("[dim]Translating...[/dim]"):
                    try: user_msg = translator.translate(user_msg, src="ro", dest="en").text
                    except: pass


            messages = [{"role": "system", "content": "You are Hen, a helpful AI assistant."}]
            messages.extend(conversation_history[-3:]) 
            messages.append({"role": "user", "content": user_msg})
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            console.print("\n[bold yellow]🐔 Hen:[/bold yellow]")
            with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="aesthetic"):
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    temperature=CURRENT_TEMP, 
                    do_sample=True, 
                    top_p=0.9
                )
            
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = raw_output.split("assistant")[-1].strip() if "assistant" in raw_output.lower() else raw_output

            # Translate Output
            if ROMANIAN_MODE:
                try: response = translator.translate(response, src="en", dest="ro").text
                except: response = f"[Translation Error] {response}"

            # Render Markdown
            md = Markdown(response)
            console.print(md)
            
            conversation_history.append({"role": "user", "content": user_msg})
            conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    select_model()
    run_chat()