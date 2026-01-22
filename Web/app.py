import os
import rich
from pathlib import Path
import json
import torch
import base64
from peft import PeftConfig, PeftModel
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from googletrans import Translator
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    BlipProcessor, BlipForConditionalGeneration
)
from rich import print as rich_print
##########################################################################################################################
# HEN WEBUI Backend V1.5                                                                                                 #
# Major Updates:                                                                                                         #
# some bugfixes,and initial message                                                                                      #
##########################################################################################################################
rich_print("[bold magenta]--Hen WebUI Backend V1.5--[/bold magenta]")
app = Flask(__name__)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_DIR = "./HenModels"
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

model = None
tokenizer = None
caption_model = None
caption_processor = None
current_model_name = "None"
# --- HELPER FUNCTIONS ---
def load_image_model():
    """Loads the image describer (BLIP) only once."""
    global caption_model, caption_processor
    if caption_model is None:
        print("👁️ Loading Image Describer...")
        caption_processor = BlipProcessor.from_pretrained(IMAGE_CAPTION_MODEL)
        caption_model = BlipForConditionalGeneration.from_pretrained(IMAGE_CAPTION_MODEL).to("cuda")
def describe_image(image_file):
    """Generates a text description of an uploaded image."""
    load_image_model()
    try:
        raw_image = Image.open(image_file).convert('RGB')
        inputs = caption_processor(raw_image, return_tensors="pt").to("cuda")
        out = caption_model.generate(**inputs)
        description = caption_processor.decode(out[0], skip_special_tokens=True)
        return f"[User uploaded an image. Description: {description}]"
    except Exception as e:
        print(f"Image Error: {e}")
        return "[Error analyzing image]"
def get_available_models():
    """Scans the HenModels directory."""
    if not os.path.exists(BASE_MODEL_DIR): return []
    models = []
    for folder in os.listdir(BASE_MODEL_DIR):
        folder_path = os.path.join(BASE_MODEL_DIR, folder)
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "adapter_config.json")):
            metadata = {"tier": "Unknown"}
            try:
                with open(os.path.join(folder_path, "hen_desc.json"), 'r') as f:
                    metadata = json.load(f)
            except: pass
            models.append({
                "id": folder_path,
                "name": folder,
                "tier": metadata.get("tier", "Custom")
            })
    return models

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/scan', methods=['GET'])
def scan_models():
    return jsonify(get_available_models())
from peft import PeftConfig, PeftModel
@app.route('/load_model', methods=['POST'])
def load_model_route():
    global model, tokenizer, current_model_name
    data = request.json
    selected_path = data.get('path')
    if model:
        del model
        torch.cuda.empty_cache()

    abs_path = os.path.abspath(selected_path)
    print(f"🐔 Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    if selected_path and selected_path != "base":
        print(f"🐔 Bypassing Validator. Loading manually from: {abs_path}")

        try:
            config = PeftConfig.from_pretrained(abs_path)
            model = PeftModel(base_model, config)
            adapters_weights = torch.load(os.path.join(abs_path, "adapter_model.bin"), map_location="cuda")
            model.load_state_dict(adapters_weights, strict=False)
        except:

            from safetensors.torch import load_file
            weights_path = os.path.join(abs_path, "adapter_model.safetensors")
            if os.path.exists(weights_path):
                adapters_weights = load_file(weights_path, device="cuda")
                model.load_state_dict(adapters_weights, strict=False)
        current_model_name = os.path.basename(selected_path)
    else:
        model = base_model
        current_model_name = "Base Qwen 2.5"
    return jsonify({"status": "success", "loaded": current_model_name})
@app.route('/chat', methods=['POST'])
def chat():
    global model, tokenizer
    if not model:
        return jsonify({"error": "No model loaded! Select one from the sidebar."}), 400
    text = request.form.get('text', '')
    romanian_mode = request.form.get('romanian') == 'true'
    image = request.files.get('image')
    image_context = ""
    if image:
        image_context = describe_image(image)
        text = f"{image_context}\n\nUser Question: {text}"

    translator = Translator()
    if romanian_mode:
        try:
            text = translator.translate(text, src='ro', dest='en').text
        except Exception as e:
            print(f"Trans Error: {e}")

    messages = [
        {"role": "system", "content": "You are Hen, a helpful AI assistant."},
        {"role": "user", "content": text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    def generate():
        streamer = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.7, 
            do_sample=True, 
            top_p=0.9
        )
        decoded_text = tokenizer.decode(streamer[0], skip_special_tokens=True)

        response = decoded_text.split("assistant")[-1].strip()

        if romanian_mode:
            try:
                response = translator.translate(response, src='en', dest='ro').text
            except: pass
        yield response
    return Response(generate(), mimetype='text/plain')
if __name__ == '__main__':
    if not os.path.exists(BASE_MODEL_DIR): os.makedirs(BASE_MODEL_DIR)
    app.run(host='0.0.0.0', port=5000, debug=True)