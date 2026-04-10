import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import json
import os
from typing import Literal
from unittest.mock import patch
from transformers import PreTrainedModel


import grpo_helper

from zero_shot_baseline import get_prompt
from zero_shot_baseline import prompt_generation

from helper_methods import tokenize_prompt_and_output
from helper_methods import get_response_log_probs
from drgrpo_grader import r1_zero_reward_fn



# create dataset
# Dataset: (prompt, ground_truth)
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

def init_vllm(model_id: str, 
        device: str, 
        seed: int, 
        gpu_memory_utilization: float = 0.85
    ):
        vllm_set_random_seed(seed)
        world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
        profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
        with world_size_patch, profiling_patch:
            return LLM(
                model=model_id,
                device=device,
                dtype=torch.bfloat16,
                enable_prefix_caching=True,
                gpu_memory_utilization=gpu_memory_utilization,
                )

# load the current policy to initiate vllm
# model evluation
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())



def grpo_train(
    dataset = None,
    model_path: str = "",
    gpu_memory_utilization: float = 0.85,
    learning_rate: float = 1e-5,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    n_grpo_steps: int = 200, # training epochs
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    train_batch_size: int = 256, # On-policy
    gradient_accumulation_steps: int = 128,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip",] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    eval_interval = 10,
    eval_dataset = None,
    save_dir = "./checkpoints"
):  
    # ===== sanity check =====
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
    
    # ===== load the dataset ======
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = n_prompts_per_rollout_batch,
        shuffle = True,
        drop_last = False
    )

    data_iter = iter(data_loader)

    # ===== set the device ======
    device = "cuda:0"

    # ===== load the data ======
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    
    # ===== load the tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    
    # ===== vllm initiation ======
    vllm_model = init_vllm(
        model_id = model_path,
        device = device,
        seed = 0,
        gpu_memory_utilization = 0.2)

    sampling_params = SamplingParams(
        temperature=sampling_temperature, 
        top_p=1.0, 
        max_tokens=sampling_max_tokens,
        min_tokens= sampling_min_tokens,
        n = group_size,
        stop=["</answer>"],
    )
    sampling_params.include_stop_str_in_output = True

    
    # ===== optimizer ======
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr =learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )


    # ===== gradient accumulation ======
    global_microbatch_step = 0
    optimizer.zero_grad()

    best_score = -float("inf") # to record the best score
    
    
    ###### start training ######
    print("Start training ... ")
    for step in range(n_grpo_steps):
        model.train()  # train mode

        
        # ===== sample prompts =====
        try:
            prompts, ground_truths = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            prompts, ground_truths= next(data_iter)
        

        # ===== build rollout data =====
        outputs = vllm_model.generate(
            prompts, 
            sampling_params
        )

        rollout_responses = [] # rollout_batch_size
        for output in outputs:
            for response in output.outputs:
                rollout_responses.append(response.text)

        rollout_ground_truths = []
        for ground_truth in ground_truths:
            rollout_ground_truths.extend([ground_truth] * group_size)

        rollout_prompts = []
        for prompt in prompts:
            rollout_prompts.extend([prompt] * group_size)

        
        # ===== compute rewards =====
        # advantages shape (rollout_batch_size,)
        # raw rewards shape (rollout_batch_size,)
        advantages, raw_rewards, _ = grpo_helper.compute_group_normalized_rewards(
            reward_fn = r1_zero_reward_fn,
            rollout_responses = rollout_responses,
            repeated_ground_truths = rollout_ground_truths,
            group_size = group_size,
            advantage_eps = advantage_eps,
            normalize_by_std = use_std_normalization,
        )

        advantages = advantages.unsqueeze(-1).to(device)
        raw_rewards = raw_rewards.unsqueeze(-1).to(device)


        # ===== tokenize =====
        tokenized = tokenize_prompt_and_output(
            prompt_strs = rollout_prompts,
            output_strs = rollout_responses ,
            tokenizer = tokenizer,
        )

        input_ids = torch.tensor(tokenized["input_ids"]).to(device)
        labels = torch.tensor(tokenized["labels"]).to(device)
        response_mask = torch.tensor(tokenized["response_mask"]).to(device)



        # ===== microbatch training  =====
        for i in range(n_microbatches_per_rollout_batch):
            start = i * micro_train_batch_size
            end = start + micro_train_batch_size

            mb_input_ids = input_ids[start:end].to(device)
            mb_labels = labels[start:end].to(device)
            mb_mask = response_mask[start:end].to(device)
            mb_advantages = advantages[start:end].to(device)
            mb_rewards = raw_rewards[start:end].to(device)

            # compute log probabilities
            response_log_probs = get_response_log_probs(
                model = model,
                input_ids = mb_input_ids,
                labels = mb_labels,
                return_token_entropy = False
            )

            mb_log_probs = response_log_probs["log_probs"]

            # old log probs
            # for clipping GRPO
            # mb_old_log_probs = mb_log_probs.detach()

            loss, _ = grpo_helper.grpo_microbatch_train_step(
                policy_log_probs = mb_log_probs,
                response_mask = mb_mask,
                gradient_accumulation_steps = gradient_accumulation_steps,
                loss_type = loss_type,
                raw_rewards = mb_rewards,
                advantages = mb_advantages,
                old_log_probs = None,
                cliprange = None,
            )


            global_microbatch_step += 1

            if global_microbatch_step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        

        # ===== logging =====
        if step < 15:
            print("raw_rewards mean:", raw_rewards.mean().item())
            print("raw_rewards unique:", torch.unique(raw_rewards))
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.10f}")



        # ===== vllm evalution ======
        if eval_dataset is not None and (step + 1) % eval_interval == 0:
            print("Running vLLM evaluation...")

            # load latest policy weights into vLLM
            load_policy_into_vllm_instance(
                policy=model,
                llm=vllm_model
            )

            eval_loader = DataLoader(
                eval_dataset,
                batch_size=n_prompts_per_rollout_batch,
                shuffle=False,
                drop_last=False
                )

            all_rewards = []

            for eval_prompts, eval_ground_truths in eval_loader:
                outputs = vllm_model.generate(
                    eval_prompts,
                    sampling_params
                    )

                eval_responses = []
                for output in outputs:
                    eval_responses.append(output.outputs[0].text)
                
                rewards = []
                for response, ground_truth in zip(eval_responses, eval_ground_truths):
                    reward = r1_zero_reward_fn(response, ground_truth)
                    rewards.append(reward)

                all_rewards.extend(rewards)

            score = sum(all_rewards) / len(all_rewards)
            print(f"[Eval] Step {step} | Score: {score:.4f}")


            # save best model
            if score > best_score:
                best_score = score
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(os.path.join(save_dir, "best_model"))
                tokenizer.save_pretrained(os.path.join(save_dir, "best_model"))
                print(f"New best model saved (score={score:.4f})")

    
    # ===== save last model =====
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(os.path.join(save_dir, "last_model"))
    tokenizer.save_pretrained(os.path.join(save_dir, "last_model"))
    print("Last model saved")

    print("Finish!")

if __name__ == "__main__":
    # model path
    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B"

    
    # train dataset = 7.5k
    data_file = "/root/autodl-tmp/assignment5-alignment/data/math_train_512.json"
    prompt_file = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    dataset = data_processing(data_file, prompt_file)

    print(dataset[0])
    print(dataset[1])

    
    # eval dataset = 5k
    data_file = "/root/autodl-tmp/assignment5-alignment/data/math_test.json"
    prompt_file = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    eval_dataset = data_processing(data_file, prompt_file)


    # start training and evluations
    grpo_train(
        dataset = dataset,
        model_path = model_path,
        loss_type = "reinforce_with_baseline",
        eval_interval = 10,
        eval_dataset = eval_dataset,
        save_dir = "./checkpoints"
    )