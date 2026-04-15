import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
BASE_MODEL_PATH = "/home/zelin/FedBiOT-master/local_models/gemma-2-2b"
# ---------------------

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    # --- Step 1: Load Model and Tokenizer ---
    print(f"\n--- Loading Tokenizer and Model from '{BASE_MODEL_PATH}' ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("\n--- Base model loaded successfully! ---")

except Exception as e:
    print(f"\n--- [ERROR] Failed to load the base model. ---")
    print(f"Detailed error: {e}")
    exit()

# --- Step 2: Format the prompt correctly ---

# FIX 1: Manually set the official Gemma chat template
gemma_template = (
    "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<start_of_turn>model\n' }}"
    "{% endif %}"
)
tokenizer.chat_template = gemma_template


messages = [
    {"role": "user", "content": "请介绍一下北京。"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n--- Prompt after applying CORRECT Gemma-2 template: ---")
print(prompt)


# --- Step 3: Generate a Response ---
print("\n--- Generating a response... ---")

# Tokenize the prompt to get 'input_ids' and 'attention_mask'
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# FIX 2: Use the **inputs unpacking to pass all required arguments
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

# --- Step 4: Decode and Print the Result ---
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n==================== Generation Result ====================")
print(generated_text)
print("=========================================================")