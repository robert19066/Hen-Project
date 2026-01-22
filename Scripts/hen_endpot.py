#!/usr/bin/env python3


import os
import json
import torch
import argparse
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import threading
import sys


MODELS_DIR = "HenModels"
CONFIG_FILE = "hen_endpoint_config.json"

MODEL_CONFIGS = {
    "standard": "Qwen/Qwen2.5-3B-Instruct",
    "reasoning": "Qwen/QwQ-32B-Preview",
    "lite": "Qwen/Qwen2.5-1.5B-Instruct",
}


app = Flask(__name__)
CORS(app)  

class ModelCache:
    def __init__(self):
        self.models = {}  
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.available_models = []
        
cache = ModelCache()
server_config = {
    "host": "0.0.0.0",
    "port": 5000,
    "custom_url": "/api/hen",
    "api_key": None,
    "max_tokens": 512,
    "temperature": 0.7
}


def scan_available_models(sort_order="new_to_old"):
    """Scan and return available models with sorting"""
    if not os.path.exists(MODELS_DIR):
        return []
    
    models = []
    
    for item in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, item)
        if not os.path.isdir(path):
            continue
        
        meta_path = os.path.join(path, "hen_metadata.json")
        
        model_info = {
            "name": item,
            "path": path,
            "tier": "UNKNOWN",
            "status": "unknown",
            "created_at": None,
            "base_model": "standard"
        }
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                    model_info["tier"] = data.get("tier", "UNKNOWN")
                    model_info["status"] = "ready" if data.get("training_steps") else "untrained"
                    model_info["created_at"] = data.get("created_at")
                    model_info["base_model"] = data.get("base_model_type", "standard")
                    model_info["description"] = data.get("description", "")
            except Exception as e:
                print(f"⚠️  Error reading metadata for {item}: {e}")
        

        if model_info["status"] == "ready":
            models.append(model_info)
    

    models_with_dates = [m for m in models if m["created_at"]]
    models_without_dates = [m for m in models if not m["created_at"]]
    
    if sort_order == "new_to_old":
        models_with_dates.sort(key=lambda x: x["created_at"], reverse=True)
    else:  # old_to_new
        models_with_dates.sort(key=lambda x: x["created_at"])
    
    return models_with_dates + models_without_dates

def load_model_by_index(model_index, sort_order="new_to_old"):
    """Load a model by its index in the sorted list"""
    models = scan_available_models(sort_order)
    
    if not models:
        raise ValueError("No trained models available")
    
    if model_index < 1 or model_index > len(models):
        raise ValueError(f"Model index {model_index} out of range (1-{len(models)})")
    
    model_info = models[model_index - 1]  
    return load_model(model_info["path"], model_info["name"])

def load_model_by_name(model_name):
    """Load a model by its name"""
    model_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model '{model_name}' not found")
    
    return load_model(model_path, model_name)

def load_model(model_path, model_name):
    """Load a specific model into memory"""
    print(f"🔄 Loading model: {model_name}...")
    
    # Check if already loaded
    if cache.current_model_name == model_name:
        print(f"✅ Model already loaded: {model_name}")
        return {
            "model": cache.current_model,
            "tokenizer": cache.current_tokenizer,
            "name": cache.current_model_name
        }
    

    meta_path = os.path.join(model_path, "hen_metadata.json")
    base_model_type = "standard"
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            base_model_type = metadata.get("base_model_type", "standard")
    
    base_model_id = MODEL_CONFIGS.get(base_model_type, MODEL_CONFIGS["standard"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    

    model = PeftModel.from_pretrained(base_model, model_path)

    cache.current_model = model
    cache.current_tokenizer = tokenizer
    cache.current_model_name = model_name
    
    print(f"✅ Loaded: {model_name}")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "name": model_name
    }

def generate_response(prompt, max_tokens=None, temperature=None):
    """Generate a response using the current model"""
    if cache.current_model is None or cache.current_tokenizer is None:
        raise ValueError("No model loaded")
    
    max_tokens = max_tokens or server_config["max_tokens"]
    temperature = temperature or server_config["temperature"]
    

    messages = [
        {"role": "system", "content": "You are Hen, a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]


    formatted_prompt = cache.current_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    

    inputs = cache.current_tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(cache.current_model.device)
    

    with torch.no_grad():
        outputs = cache.current_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=cache.current_tokenizer.eos_token_id
        )
    

    response = cache.current_tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def check_api_key():
    """Check if API key is required and valid"""
    if server_config["api_key"] is None:
        return True
    
    provided_key = request.headers.get("X-API-Key") or request.args.get("api_key")
    return provided_key == server_config["api_key"]

@app.route(f'{server_config["custom_url"]}', methods=['POST'])
def api_endpoint():
    """Main API endpoint"""
    

    if not check_api_key():
        return jsonify({"error": "Invalid or missing API key"}), 401
    
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        action = data.get("action")
        

        if action == "list":
            sort_order = data.get("modelIndexSort", "new_to_old")
            models = scan_available_models(sort_order)
            
            return jsonify({
                "success": True,
                "action": "list",
                "count": len(models),
                "models": [
                    {
                        "index": idx + 1,
                        "name": m["name"],
                        "tier": m["tier"],
                        "created_at": m["created_at"],
                        "description": m.get("description", "")
                    }
                    for idx, m in enumerate(models)
                ],
                "sort_order": sort_order
            })
        

        elif action == "run":
            container = data.get("container")
            
            if not container:
                return jsonify({"error": "No prompt provided in 'container'"}), 400
            

            model_name = data.get("model")
            model_index = data.get("modelIndex")
            sort_order = data.get("modelIndexSort", "new_to_old")
            

            if model_index is not None:
                load_model_by_index(model_index, sort_order)
            elif model_name:
                load_model_by_name(model_name)
            else:

                if cache.current_model is None:
                    return jsonify({"error": "No model specified and no model currently loaded"}), 400
            

            max_tokens = data.get("max_tokens", server_config["max_tokens"])
            temperature = data.get("temperature", server_config["temperature"])
            

            response = generate_response(container, max_tokens, temperature)
            
            return jsonify({
                "success": True,
                "action": "run",
                "model": cache.current_model_name,
                "prompt": container,
                "response": response,
                "tokens_used": len(cache.current_tokenizer.encode(response))
            })

        elif action == "load":
            model_name = data.get("model")
            model_index = data.get("modelIndex")
            sort_order = data.get("modelIndexSort", "new_to_old")
            
            if model_index is not None:
                result = load_model_by_index(model_index, sort_order)
            elif model_name:
                result = load_model_by_name(model_name)
            else:
                return jsonify({"error": "No model specified"}), 400
            
            return jsonify({
                "success": True,
                "action": "load",
                "model": result["name"],
                "message": f"Model {result['name']} loaded successfully"
            })

        elif action == "status":
            return jsonify({
                "success": True,
                "action": "status",
                "current_model": cache.current_model_name,
                "server_config": {
                    "custom_url": server_config["custom_url"],
                    "max_tokens": server_config["max_tokens"],
                    "temperature": server_config["temperature"],
                    "api_key_enabled": server_config["api_key"] is not None
                },
                "available_models": len(scan_available_models())
            })
        
        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "server": "Hen-2B API Endpoint",
        "current_model": cache.current_model_name
    })


def load_config():
    """Load server configuration"""
    global server_config
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded = json.load(f)
                server_config.update(loaded)
            print(f"✅ Loaded config from {CONFIG_FILE}")
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")

def save_config():
    """Save server configuration"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(server_config, f, indent=2)
        print(f"✅ Saved config to {CONFIG_FILE}")
    except Exception as e:
        print(f"⚠️  Error saving config: {e}")


def print_header():
    """Print CLI header"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("╔" + "═" * 68 + "╗")
    print("║" + " 🐔 HEN ENDPOT SERVER ".center(67) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + " Turn your Hen models into a custom API service! ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

def interactive_menu():
    """Interactive configuration menu"""
    print_header()
    
    while True:
        print("\n📋 CURRENT CONFIGURATION:")
        print(f"  • Host:        {server_config['host']}")
        print(f"  • Port:        {server_config['port']}")
        print(f"  • Custom URL:  {server_config['custom_url']}")
        print(f"  • API Key:     {'Set' if server_config['api_key'] else 'Not Set'}")
        print(f"  • Max Tokens:  {server_config['max_tokens']}")
        print(f"  • Temperature: {server_config['temperature']}")
        
        models = scan_available_models()
        print(f"\n🐔 Available Models: {len(models)}")
        for idx, model in enumerate(models, 1):
            print(f"  {idx}. {model['name']} [{model['tier']}]")
        
        print("\n⚙️  OPTIONS:")
        print("  1. Configure Server Settings")
        print("  2. Set API Key (optional)")
        print("  3. Test Model Loading")
        print("  4. Start Server")
        print("  5. Exit")
        
        choice = input("\n→ Select option: ").strip()
        
        if choice == '1':
            configure_server()
        elif choice == '2':
            configure_api_key()
        elif choice == '3':
            test_model_loading()
        elif choice == '4':
            start_server()
            break
        elif choice == '5':
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice!")

def configure_server():
    """Configure server settings"""
    print("\n🔧 SERVER CONFIGURATION\n")
    
    host = input(f"Host (current: {server_config['host']}): ").strip()
    if host:
        server_config['host'] = host
    
    port = input(f"Port (current: {server_config['port']}): ").strip()
    if port:
        try:
            server_config['port'] = int(port)
        except:
            print("❌ Invalid port number!")
    
    url = input(f"Custom URL path (current: {server_config['custom_url']}): ").strip()
    if url:
        if not url.startswith('/'):
            url = '/' + url
        server_config['custom_url'] = url

        app.add_url_rule(url, 'api_endpoint', api_endpoint, methods=['POST'])
    
    max_tok = input(f"Max tokens (current: {server_config['max_tokens']}): ").strip()
    if max_tok:
        try:
            server_config['max_tokens'] = int(max_tok)
        except:
            print("❌ Invalid number!")
    
    temp = input(f"Temperature (current: {server_config['temperature']}): ").strip()
    if temp:
        try:
            server_config['temperature'] = float(temp)
        except:
            print("❌ Invalid number!")
    
    save_config()
    print("\n✅ Configuration saved!")

def configure_api_key():
    """Configure API key"""
    print("\n🔐 API KEY CONFIGURATION\n")
    print("Leave empty to disable API key authentication.")
    
    key = input("Enter API key: ").strip()
    
    if key:
        server_config['api_key'] = key
        print(f"\n✅ API key set! Include it in requests:")
        print(f"   Header: X-API-Key: {key}")
        print(f"   Or URL: ?api_key={key}")
    else:
        server_config['api_key'] = None
        print("\n✅ API key authentication disabled")
    
    save_config()

def test_model_loading():
    """Test loading a model"""
    print("\n🧪 MODEL LOADING TEST\n")
    
    models = scan_available_models()
    if not models:
        print("❌ No trained models available!")
        input("\nPress Enter to continue...")
        return
    
    print("Available models:")
    for idx, model in enumerate(models, 1):
        print(f"  {idx}. {model['name']} [{model['tier']}]")
    
    choice = input("\nSelect model number: ").strip()
    
    try:
        idx = int(choice)
        load_model_by_index(idx)
        print(f"\n✅ Successfully loaded: {cache.current_model_name}")
        

        test = input("\nTest generation? (y/n): ").strip().lower()
        if test == 'y':
            prompt = input("Enter test prompt: ").strip()
            if prompt:
                print("\n🤔 Generating...")
                response = generate_response(prompt, max_tokens=100)
                print(f"\n🐔 Response:\n{response}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")

def start_server():
    """Start the Flask server"""
    print("\n🚀 STARTING SERVER...\n")
    
    print(f"🌐 API Endpoint: http://{server_config['host']}:{server_config['port']}{server_config['custom_url']}")
    print(f"❤️  Health Check: http://{server_config['host']}:{server_config['port']}/health")
    
    if server_config['api_key']:
        print(f"🔐 API Key Required: {server_config['api_key']}")
    
    print("\n📖 USAGE EXAMPLES:")
    print(f"""
# List all models
curl -X POST http://localhost:{server_config['port']}{server_config['custom_url']} \\
  -H "Content-Type: application/json" \\
  -d '{{"action": "list"}}'

# Run inference with model by index
curl -X POST http://localhost:{server_config['port']}{server_config['custom_url']} \\
  -H "Content-Type: application/json" \\
  -d '{{"action": "run", "modelIndex": 1, "container": "Hello, how are you?"}}'

# Run with specific model name
curl -X POST http://localhost:{server_config['port']}{server_config['custom_url']} \\
  -H "Content-Type: application/json" \\
  -d '{{"action": "run", "model": "my_model", "container": "Write a poem"}}'
""")
    
    print("="  * 70)
    print("\n🐔 Server is running! Press Ctrl+C to stop.\n")
    
    app.run(
        host=server_config['host'],
        port=server_config['port'],
        debug=False,
        threaded=True
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="🐔 Hen Endpot Server")
    parser.add_argument('--host', type=str, help='Server host')
    parser.add_argument('--port', type=int, help='Server port')
    parser.add_argument('--no-menu', action='store_true', help='Skip interactive menu')
    
    args = parser.parse_args()
    

    load_config()
    

    if args.host:
        server_config['host'] = args.host
    if args.port:
        server_config['port'] = args.port
    
    if args.no_menu:
        start_server()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()