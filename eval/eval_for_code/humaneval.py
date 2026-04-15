import os
import torch
import json
import transformers
from transformers import GenerationConfig
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

DEBUG = False
NUM_ANSWERS_PER_QUESTION = 5


def clean_answer(code):
    """
    Cleans the generated code to make it executable.
    """
    def pad_spaces(s, num=4):
        n = 0
        while n < len(s) and s[n] == " ":
            n += 1
        if n != num:
            s = " " * num + s[n:]
        return s

    # 1. Handle markdown code blocks
    if "```" in code:
        code = code.split("```")[0]

    # 2. Replace special unicode spaces
    code = code.replace('\u2581', ' ')
    code = code.replace('\u00a0', ' ')

    # 3. Remove explanations and other stop sequences
    stop_sequences = [
        '\nclass', '\ndef', '\n#', '\nif', '\nprint', '\nassert',
        '**Explanation**', 'Explanation:'
    ]
    for stop_seq in stop_sequences:
        code = code.split(stop_seq)[0]

    # 4. Remove any trailing whitespace and pad
    code = code.rstrip()
    code = pad_spaces(code, 4)
    
    return code


@torch.no_grad()
def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)

    # Get test file
    fp = os.path.join('./data/HumanEval.jsonl.gz')
    if not os.path.exists(fp):
        download_url(
            'https://github.com/openai/human-eval/raw/'
            '463c980b59e818ace59f6f9803cd92c749ceae61/'
            'data/HumanEval.jsonl.gz', init_cfg.data.root)
    list_data_dict = load_jsonl(fp,
                                instruction='prompt',
                                input='entry_point',
                                category='task_id',
                                output='test',
                                is_gzip=True)
        # 在这里定义您的少样本示例
    FEW_SHOT_EXAMPLES = """
Q:
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
A:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False

Q:
from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return them in a list.
    Separate groups are balanced parentheses left next to each other (and not inside each other).
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"
A:
    result = []
    current_string = ""
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string += c
        elif c == ')':
            current_depth -= 1
            current_string += c
            if current_depth == 0:
                result.append(current_string)
                current_string = ""

    return result

"""

    while True:
        try:
            out_file = os.path.join(
                init_cfg.outdir, f'{fschatbot.curpfx}humaneval_answer.jsonl')
            answers = []
            for sample in tqdm(list_data_dict):
                # 将少样本示例和实际任务指令结合起来
                prompt = f"{FEW_SHOT_EXAMPLES}Q:\n{sample['instruction']}\nA:"
                input_text = prompt  # 使用新的 prompt 作为输入

                generation_config = GenerationConfig(
                    temperature=0.1,
                    top_k=40,
                    top_p=0.75,
                    do_sample=True,
                    num_return_sequences=NUM_ANSWERS_PER_QUESTION,
                )
                generate_kwargs = dict(
                    generation_config=generation_config,
                    max_new_tokens=128,
                )
                try:
                    model_completions = fschatbot.generate(
                        input_text, generate_kwargs)
                except torch.cuda.OutOfMemoryError() as error:
                    print(error)
                    model_completions = [
                        '' for _ in range(NUM_ANSWERS_PER_QUESTION)
                    ]

                for i, completion in enumerate(model_completions):
                    completion = clean_answer(completion)
                    answers.append(
                        dict(task_id=sample['category'],
                             completion=completion))
                    if DEBUG:
                        print(f"task_id: {sample['category']},\n"
                              f"completion {i + 1}:\n{completion}\n\n")
            # Save as samples.jsonl for eval pass@k score
            # Run `evaluate_functional_correctness samples.jsonl`
            with open(out_file, 'w') as f:
                for answer in answers:
                    json_str = json.dumps(answer)
                    f.write(json_str + '\n')

            print('load the next model...')
            fschatbot.next_model()
        except Exception as err:
            print(f'{err}, so finished all evaluations....')
            break


if __name__ == "__main__":
    main()
