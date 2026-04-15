from federatedscope.llm.model.adapter_builder import AdapterModel
import torch
import time
from transformers import AutoModelForCausalLM, GPTQConfig


def get_model_from_huggingface(model_name, config, **kwargs):
    from transformers import AutoModelForCausalLM

    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    # kwargs['attn_implementation'] = "flash_attention_2"

    #当使用tinyllama时，一定要启用；使用llama2时不启用
    kwargs['ignore_mismatched_sizes'] = True

    if config.llm.gptq.use:
        print("Loading model with AutoGPTQ.")
        gptq_config = GPTQConfig(
            bits=config.llm.gptq.bits,
            dataset=config.llm.gptq.dataset,
            tokenizer=model_name, # GPTQ需要tokenizer来处理校准数据
            damp_percent=config.llm.gptq.damp_percent,
            desc_act=config.llm.gptq.desc_act,
            sym=config.llm.gptq.sym,
            true_sequential=config.llm.gptq.true_sequential,
            use_cuda_fp16=config.llm.gptq.use_cuda_fp16,
            # Fix: Use the token length from config for model sequence length
            model_seqlen=config.llm.tok_len
        )
        kwargs['quantization_config'] = gptq_config
        # 对于量化模型，device_map是必需的
        kwargs['device_map'] = 'auto'
        print("Quantizing model... This may take a while.")

        # from_pretrained 会自动处理量化过程
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        print("Model quantization finished.")
        return model
    if config.train.is_enable_half:
        kwargs['torch_dtype'] = torch.bfloat16
        if config.use_gpu:
            kwargs['device_map'] = f'cuda:{config.device}'

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    return model


def get_model_from_modelscope(model_name, config, **kwargs):
    from modelscope import AutoModelForCausalLM

    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_llm(config, **kwargs):
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config,
                                           **kwargs)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name,
                                          config=config,
                                          **kwargs)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # Resize LLM model based on settings
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)

    return model