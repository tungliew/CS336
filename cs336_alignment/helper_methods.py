##### SFT Helper Methods #####
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    '''
    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.
    '''
    prompt_tokens = [tokenizer(prompt)["input_ids"] for prompt in prompt_strs]
    output_tokens = [tokenizer(output)["input_ids"] for output in output_strs]

    input_ids = []
    labels = []
    response_mask = []

    tokens_len = [len(prompt)+len(output) for prompt, output in zip(prompt_tokens, output_tokens)]
    max_len = max(tokens_len)

    pad_token_id = tokenizer.pad_token_id

    for prompt, output in zip(prompt_tokens, output_tokens):
        full_token = prompt + output

        pad_len = max_len - len(full_token)
        full_token = full_token + [pad_token_id] * pad_len

        input_id = full_token[:-1]
        label = full_token[1:]

        mask = [0] * (len(prompt)-1) + [1]*len(output) + [0]*pad_len
        
        input_ids.append(input_id)
        labels.append(label)
        response_mask.append(mask)
    

    output = {"input_ids": input_ids,
              "labels": labels,
              "response_mask": response_mask}

    return output



def compute_entropy(logits):
    '''
    Args:
    logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size) 
    containing unnormalized logits.
    '''

    max_val = torch.max(logits, dim=-1, keepdim=True).values
    
    logits_sub = logits - max_val
    logits_exp = torch.exp(logits_sub)
    z = torch.sum(logits_exp, dim=-1, keepdim=True)
    log_z = torch.log(z) + max_val

    # probs = torch.exp(logits - log_z)
    probs = logits_exp / z

    entropy = log_z.squeeze(-1) - torch.sum(probs * logits, dim=-1)

    return entropy


def get_response_log_probs(
        model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy = False
):
    logits = model(input_ids).logits
    logits_softmax = F.log_softmax(logits, dim=-1)
    log_probs = logits_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    output = {"log_probs": log_probs}

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        output["token_entropy"] = token_entropy
    
    
    return output



def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None= None,
):
    mask = mask.to(dtype = tensor.dtype)
    tensor_masked = tensor * mask
    output = torch.sum(tensor_masked, dim= dim) / normalize_constant
    return output


# FAILED
def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
):
    '''
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the SFT policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
    normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    '''
    
    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=None)
    
    loss = -loss
    
    loss = loss / gradient_accumulation_steps

    loss.backward()

    metadata = {}

    return loss, metadata
