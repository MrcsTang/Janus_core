import os
import re
import transformers
import json
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

INVALID_ANS = "[invalid]"

def build_prompt(instruction, input_text):
    """
    根据 instruction (问题) 和 input (上下文) 构建 Prompt。
    注意：这里的模板需要与你训练 (SFT) 时使用的模板保持一致。
    """
    prompt = (
        "### Instruction:\n"
        "Please answer the question based on the following context. "
        "Your answer must end with 'yes', 'no', or 'maybe'.\n\n"
        f"Context: {input_text}\n\n"
        f"Question: {instruction}\n\n"
        "### Response:\n"
    )
    return prompt

def extract_gt_answer(output_text):
    """
    从数据集的 output 字段中提取真实的标签 (yes/no/maybe)。
    针对格式: "... \nConclusion: no"
    """
    # 优先匹配 "Conclusion: xxx" 的格式
    match = re.search(r'Conclusion:\s*(yes|no|maybe)', output_text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    
    # 作为 fallback，寻找文本中最后一个出现的 yes, no 或 maybe
    matches = re.findall(r'\b(yes|no|maybe)\b', output_text.lower())
    if matches:
        return matches[-1]
    
    return INVALID_ANS

def clean_model_answer(model_pred):
    """
    从模型生成的文本中提取 yes, no, 或 maybe。
    由于模型可能会输出一段推理过程，我们尽量取它最后得出的结论。
    """
    model_pred = model_pred.lower().strip()
    
    # 优先匹配模型是否按照格式输出了 "Conclusion: xxx"
    match = re.search(r'conclusion:\s*(yes|no|maybe)', model_pred)
    if match:
        return match.group(1)

    # 否则，提取文本中最后出现的一个分类词
    matches = re.findall(r'\b(yes|no|maybe)\b', model_pred)
    if matches:
        return matches[-1] # 通常结论在最后
    else:
        return INVALID_ANS

def is_correct(model_answer, gt_answer):
    if gt_answer == INVALID_ANS or model_answer == INVALID_ANS:
        return False
    return model_answer == gt_answer

def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)
    init_cfg.freeze()

    # 初始化模型
    fschatbot = FSChatBot(init_cfg)

    # 读取测试文件 (假设文件名是 pubmedqa_test.jsonl 或 .json)
    fp = os.path.join(init_cfg.data.root, 'pubmedqa_output_processed.json') # 或者 .jsonl 根据你的实际情况
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Test file not found at {fp}")

    with open(fp, 'r', encoding='utf-8') as f:
        list_data_dict = json.load(f)

    if not os.path.exists(init_cfg.outdir):
        os.makedirs(init_cfg.outdir)
    results_display = open(os.path.join(init_cfg.outdir, 'pubmedqa_test_results.txt'), 'w')
    
    answers_correct = []
    testset = tqdm(list_data_dict)
    
    for sample in testset:
        # 根据你的 JSON 格式读取字段
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output_text = sample.get('output', '')

        # 提取真实的基准答案 (Ground Truth)
        gt_answer = extract_gt_answer(output_text)

        # 构建输入 Prompt
        input_prompt = build_prompt(instruction, input_text)
        
        generate_kwargs = dict(max_new_tokens=64, top_p=0.95, temperature=0.1)
        model_completion = fschatbot.generate(input_prompt, generate_kwargs)
        
        # 提取模型答案并判题
        model_answer = clean_model_answer(model_completion)
        is_cor = is_correct(model_answer, gt_answer)
        answers_correct.append(is_cor)

        results_display.write(
            f'Question (Instruction): {instruction}\n'
            f'Context (Input): {input_text[:100]}... [Truncated]\n'
            f'Original Output: {output_text}\n'
            f'--> Extracted Ground Truth: {gt_answer}\n\n'
            f'Model Full Completion: {model_completion}\n'
            f'--> Extracted Model Answer: {model_answer}\n\n'
            f'Is correct: {is_cor}\n'
            f'==========================\n\n'
        )
        results_display.flush()
        
        testset.set_postfix({
            'correct': sum(answers_correct),
            'acc': '{:.2f}%'.format(float(sum(answers_correct)) / len(answers_correct) * 100)
        })

    acc = float(sum(answers_correct)) / len(answers_correct) if len(answers_correct) > 0 else 0.0
    summary = f'Num of total questions: {len(answers_correct)}, Correct num: {sum(answers_correct)}, Accuracy: {acc:.4f}.'
    print(f"\n{summary}")
    
    results_display.write(summary + '\n')
    results_display.flush()
    results_display.close()

if __name__ == "__main__":
    main()