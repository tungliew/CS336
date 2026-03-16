import os
import json

from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn

# Data processing
# merge MATH dataset recursively
# train set 7500, test set 5000
def data_processing(folder, output_file):
    all_data = []
    with open(output_file, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(folder):
            for file in files:
                filepath = os.path.join(root, file)

                with open(filepath, "r", encoding="utf-8") as infile:
                    data = json.load(infile)
                    all_data.append(data)
                infile.close()

        json.dump(all_data, f, indent=4, ensure_ascii=False)       
    
    f.close()
    print(len(all_data))   


def prompt_generation(prompt, data):
    prompts = []
    ground_truths =  []
    for item in data:
        question = item["problem"]
        solution = item["solution"]
        generated_prompt = prompt.replace("{question}", "{"+question+"}")
        prompts.append(generated_prompt)
        ground_truths.append(solution)
    
    return (prompts, ground_truths)

def get_prompt(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()
    
    return prompt



def evaluate_vllm(
        vllm_model,
        reward_fn,
        prompts,
        ground_truths,
        eval_sampling_params,
        output_file
):
    llm = vllm_model
    outputs = llm.generate(prompts, eval_sampling_params)
    
    responses = [output.outputs[0].text for output in outputs]

    results = []
    for prompt, response, ground_truth in zip(prompts, responses, ground_truths):
        reward = reward_fn(response, ground_truth)  
        data = {"prompt":prompt,
                "response": response,
                "solution": ground_truth,
                'reward': reward}
        results.append(data)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)   
    
    f.close()



if __name__=="__main__":
    data_file = "/root/autodl-tmp/assignment5-alignment/data/math_test.json"
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    
    prompt_file = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    r1_prompt = get_prompt(prompt_file)

    prompts, ground_truths = prompt_generation(r1_prompt, data)

    eval_sampling_params = sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    
    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B"
    vllm_model = LLM(model_path)

    reward_fn = r1_zero_reward_fn

    output_file = "./result/zero_shot_baseline_output.json"

    evaluate_vllm(
        vllm_model,
        reward_fn,
        prompts,
        ground_truths,
        eval_sampling_params,
        output_file
    )

    print("successful!")