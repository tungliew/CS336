import torch
from typing import Literal
from torch.utils.data import Dataset

def compute_group_normalized_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
        group_size,
        advantage_eps,
        normalize_by_std,
):
    '''
    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).
    '''
    rollout_batch_size = len(rollout_responses)

    advantages = torch.tensor([])
    raw_rewards = torch.tensor([])

    for i in range(rollout_batch_size):
        if (i+1) % group_size == 1:
            rewards = []
        
        response = rollout_responses[i]
        ground_truth = repeated_ground_truths[i]
        reward = reward_fn(response, ground_truth)["reward"]
        
        rewards.append(reward)

        if (i+1) % group_size==0:
            rewards = torch.tensor(rewards)
            raw_rewards = torch.cat((raw_rewards, rewards))
            advantage = rewards - torch.mean(rewards)
            

            if normalize_by_std:
                advantage = advantage / (torch.std(rewards) + advantage_eps)
            
            advantages = torch.cat((advantages, advantage))
        
        
        metadata = {}
        
    
    return (advantages, raw_rewards, metadata)




def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
):
    '''
    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
    '''
    # (batch_size, sequence_length)
    naive_loss = - raw_rewards_or_advantages * policy_log_probs
    return naive_loss



def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
):
    '''
    Args:
    advantages: torch.Tensor of shape (batch_size, 1): 
        the advantages for each rollout response.
    policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
        the log-probs of the policy.
    old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
        the log-probs of the old policy.
    cliprange: float, the clip range for the ratio.
    '''
    policy_ratio = torch.exp(policy_log_probs - old_log_probs)

    clip_policy_ratio = torch.clamp(
        policy_ratio,
        min = 1.0 - cliprange,
        max = 1.0 + cliprange
    )

    # (batch_size, sequence_length)
    loss = - torch.min(policy_ratio * advantages,
                       clip_policy_ratio * advantages)
    
    metadata = {}

    return loss, metadata



def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None= None,
        advantages: torch.Tensor | None= None,
        old_log_probs: torch.Tensor | None= None,
        cliprange: float | None= None,
):
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        loss, _ = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    
    metadata = {}
    return loss, metadata



def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None= None,
):
    '''
    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.
    '''
    masked_tensor = tensor * mask
    
    if dim is None:
        total = masked_tensor.sum()
        count = mask.sum()
    else:
        total = masked_tensor.sum(dim=dim)
        count = mask.sum(dim=dim)
    
    masked_mean = total / count
    
    return masked_mean



def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None= None,
        advantages: torch.Tensor | None= None,
        old_log_probs: torch.Tensor | None= None,
        cliprange: float | None= None,
):
    per_token_loss, _ = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)

    loss = per_example_loss.mean() / gradient_accumulation_steps

    loss.backward()

    metadata = {}

    return loss, metadata 


class PromptDataset(Dataset):
    def __init__(self, prompts, ground_truths):
        self.prompts = prompts
        self.ground_truths = ground_truths
    
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx], self.ground_truths[idx]
