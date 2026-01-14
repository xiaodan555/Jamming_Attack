import argparse
import torch
import jamming_utils
import ollama

# Mock args
class Args:
    llm_model = 'qwen3:8b'
    max_response_len = 512
    seed = 0
    cache_dir = './cache'

args = Args()

# Load model/template
print("Loading model/template...")
llm_model, llm_params, conv_template = jamming_utils.load_llm_model(args, num_avail_gpus=0)

# Create a dummy context and query
context_str = "France provided money, troops, armament, military leadership, and naval support that tipped the balance of military power in favor of the United States and paved the way for the Continental Army's ultimate victory."
query_str = "why did france decide to aid the united states in its war for independence"

# Get prompt
print("Generating prompt...")
prompt = jamming_utils.get_prompt(args, conv_template, context_str, query_str)
print(f"Prompt content:\n---\n{prompt}\n---")

# Try generating
print("Calling Ollama...")
try:
    response = ollama.generate(
        model=llm_params['model'],
        prompt=prompt,
        options={'temperature': 0.0, 'num_predict': llm_params.get('max_tokens', 128)}
    )
    print(f"Response object: {response}")
    print(f"Response text: '{response['response']}'")
except Exception as e:
    print(f"Error: {e}")
