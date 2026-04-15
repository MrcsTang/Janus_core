import transformers
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.llm.misc.fschat import FSChatBot

# Import necessary functions from your original script
# You might need to adjust the import path if you save this file elsewhere
from federatedscope.llm.eval.eval_for_gsm8k.eval import create_demo_text, PROMPT_DICT

# Suppress verbose logging from transformers
transformers.logging.set_verbosity(40)


CONFIG_FILE_PATH = "/home/zelin/FedBiOT-master/federatedscope/fedbiot_gemma2_gsm8k.yaml"
# -----------------------------------------------------------------


def main():
    print(f"--- Loading configuration from: {CONFIG_FILE_PATH} ---")
    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(CONFIG_FILE_PATH)
    setup_seed(init_cfg.seed)
    init_cfg.freeze()
    print("--- Configuration loaded successfully. ---")


    print("\n--- Initializing FSChatBot to load the model... ---")
    try:
        fschatbot = FSChatBot(init_cfg)
        print("--- Model loaded successfully via FSChatBot! ---")
    except Exception as e:
        print("\n--- [ERROR] Failed to load the model via FSChatBot. ---")
        print("Please check the model configuration in your YAML file.")
        print(f"Detailed error: {e}")
        return


    print("\n--- Building a simple prompt for the smoke test... ---")
    
    simple_instruction = "请介绍一下北京。"
    
    prompt = PROMPT_DICT.get("prompt_no_input", 
                             "### Instruction:\n{instruction}\n\n### Response:").format(
                                 instruction=simple_instruction
                             )
    
    print(f"\n--- Final prompt being sent to the model: ---\n{prompt}")


    print("\n--- Generating a response... ---")
    # Using deterministic generation for debugging
    generate_kwargs = dict(max_new_tokens=256, temperature=0.0, do_sample=False)
    
    model_completion = fschatbot.generate(prompt, generate_kwargs)


    print("\n==================== Generation Result ====================")
    print(model_completion)
    print("=========================================================")


if __name__ == "__main__":
    if 'PROMPT_DICT' not in globals():
        print("[Warning] PROMPT_DICT not found. Creating a default one.")
        PROMPT_DICT = {
            "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:"
        }
    main()