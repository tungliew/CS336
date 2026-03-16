import torch
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F

from cs336_alignment.helper_methods import tokenize_prompt_and_output
from cs336_alignment.helper_methods import get_response_log_probs


# some basic code
# not tested yet
# very likely doesn't work
def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
):
    '''
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.
    '''
    end_of_sequence_token = tokenizer("</answer>")["input_ids"]
    chosen_tokens  = tokenize_prompt_and_output(prompt, response_chosen, tokenizer)
    rejected_tokens = tokenize_prompt_and_output(prompt, response_rejected, tokenizer)

    res_chosen_policy = get_response_log_probs(lm, chosen_tokens["input_ids"], chosen_tokens["labels"])
    res_rejected_policy = get_response_log_probs(lm, rejected_tokens["input_ids"], rejected_tokens["labels"])

    res_chosen_ref = get_response_log_probs(lm_ref, chosen_tokens["input_ids"], chosen_tokens["labels"])
    res_rejected_ref = get_response_log_probs(lm_ref, rejected_tokens["input_ids"], rejected_tokens["labels"])

    policy_ratio = torch.exp(res_chosen_policy - res_rejected_policy)
    ref_policy_ratio = torch.exp(res_chosen_ref - res_rejected_ref)

    loss = - torch.log(F.sigmoid(beta * (policy_ratio - ref_policy_ratio)))

    return loss

    
