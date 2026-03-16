import torch
from typing import Literal
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams


from zero_shot_baseline import get_prompt
from zero_shot_baseline import prompt_generation

import grpo_helper
from helper_methods import tokenize_prompt_and_output
from helper_methods import get_response_log_probs
from drgrpo_grader import r1_zero_reward_fn




def data_processing(data_file, prompt_file):
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    f.close()
    
    prompt = get_prompt(prompt_file)

    # r1_prompt + question
    # ground_truth - for evaluation
    # dtype: str
    prompts, ground_truths = prompt_generation(prompt, data)

    dataset = grpo_helper.PromptDataset(prompts, ground_truths)

    return dataset

    


def grpo_train(
        dataset = None,
        n_grpo_steps: int = 200,
        
        model_path: str = "",
        learning_rate: float = 1e-5,

        train_batch_size: int = 256, # On-policy, number of samples per optimization step
        gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100
        rollout_batch_size: int = 256,
        group_size: int = 8,
        epochs_per_rollout_batch: int = 1, # On-policy,

        advantage_eps: float = 1e-6,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
        use_std_normalization: bool = True,

        sampling_temperature: float = 1.0,
        sampling_min_tokens: int = 4, # As in Expiter, disallow empty string responses
        sampling_max_tokens: int = 1024,

        gpu_memory_utilization: float = 0.85,
):  
    
    ######################################
    # sanity check asserts and constants #
    ######################################
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
        )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    

    ######################################
    # initilizations                     #
    ######################################
    #load the data
    dataloader = DataLoader(
        dataset,
        batch_size = n_prompts_per_rollout_batch,
        shuffle = True
    )
    data_iter = iter(dataloader)

    # load the vllm for generation
    sampling_params = SamplingParams(
        n = group_size,
        temperature=sampling_temperature, 
        top_p=1.0, 
        max_tokens = sampling_max_tokens,
        min_tokens = sampling_min_tokens,
        stop=["/answer"],
        include_stop_str_in_output=True
    )

    vllm = LLM(model = model_path)
    
    
    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95)
    )

    ######################################
    # start training ...                 #
    ######################################
    print("start training ...")
    model.train()

    accum_step = 0

    for step in range(n_grpo_steps):
        try:
            prompts, ground_truths = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            prompts, ground_truths = next(data_iter)

        # len(responses) = len(prompts) * group_size
        # rollout_batch_size = len(responses)
        outputs = vllm.generate(prompts, sampling_params)
        
        # construct prompts, rollout_responses, repeated_ground_truths
        rollout_prompts = []
        rollout_responses = []
        repeated_ground_truths = []
        for idx, output in enumerate(outputs):
            responses = [o.text for o in output.outputs]
            rollout_responses.extend(responses)
            rollout_prompts.extend(prompts[idx] * group_size)
            repeated_ground_truths.extend(ground_truths[idx] * group_size)
        
        # compute rwards and advantages
        advantages, raw_rewards, _ = grpo_helper.compute_group_normalized_rewards(
            reward_fn = r1_zero_reward_fn,
            rollout_responses = rollout_responses,
            repeated_ground_truths = repeated_ground_truths,
            group_size = group_size,
            advantage_eps = advantage_eps,
            normalize_by_std = use_std_normalization,
        )

        # tokenization
        tokenize_output = tokenize_prompt_and_output(
            prompt_strs = rollout_prompts, 
            output_strs = rollout_responses, 
            tokenizer = tokenizer
        )

        input_ids = torch.tensor(tokenize_output["input_ids"])
        labels = torch.tensor(tokenize_output["labels"])
        response_mask = torch.tensor(tokenize_output["response_mask"])

        
        optimizer.zero_grad()

        accum_step = 0
        
        for epoch in range(epochs_per_rollout_batch):
            
            for micro_idx in range(n_microbatches_per_rollout_batch):
                start = micro_idx * micro_train_batch_size
                end = start + micro_train_batch_size

                micro_input_ids = input_ids[start:end]
                micro_labels = labels[start:end]
                micro_mask = response_mask[start:end]
                micro_advantages = advantages[start:end]

                # forward pass
                response_log_probs = get_response_log_probs(
                    model = model,
                    input_ids = micro_input_ids,
                    labels = micro_labels,
                    return_token_entropy = False
                )

                loss, _ = grpo_helper(
                    policy_log_probs = response_log_probs,
                    response_mask = micro_mask,
                    gradient_accumulation_steps = gradient_accumulation_steps,
                    loss_type =  "reinforce_with_baseline",
                    raw_rewards = raw_rewards,
                    advantages = advantages,
                    old_log_probs = None,
                    cliprangeNone = None,
                )

                accum_step += 1

                if accum_step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    # save the last model
    save_dir = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/result/grpo_model"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir) 

    print("finish training!")


if __name__ == "__main__":
    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B"

    data_file = "/root/autodl-tmp/assignment5-alignment/data/math_train.json"
    prompt_file = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    dataset = data_processing(data_file, prompt_file)

    grpo_train(
        dataset = dataset,
        model_path = model_path
    )