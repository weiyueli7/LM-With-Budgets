import argparse
import json
from inference import *

def main():
    parser = argparse.ArgumentParser(description="Generate text using a pretrained language model.")

    # Adding arguments for model directories
    parser.add_argument("--model_small_dir", type=str, default="TODO", help="Directory of the pretrained small model")
    parser.add_argument("--model_medium_dir", type=str, default="TODO", help="Directory of the pretrained medium model")
    parser.add_argument("--model_large_dir", type=str, default="TODO", help="Directory of the pretrained large model")
    parser.add_argument("--input_json", type=str, default="prompts.json", help="Input JSON file with prompts and max_lengths")

    args = parser.parse_args()

    # Load models and tokenizers
    tokenizer_small = AutoTokenizer.from_pretrained(args.model_small_dir)
    model_small = AutoModelForCausalLM.from_pretrained(args.model_small_dir)
    tokenizer_medium = AutoTokenizer.from_pretrained(args.model_medium_dir)
    model_medium = AutoModelForCausalLM.from_pretrained(args.model_medium_dir)
    tokenizer_large = AutoTokenizer.from_pretrained(args.model_large_dir)
    model_large = AutoModelForCausalLM.from_pretrained(args.model_large_dir)

    # Read input JSON file
    with open(args.input_json, 'r') as file:
        data_points = json.load(file)

    output_data = []

    for data in data_points:
        prompt = data['prompt']
        max_length = data['max_length']

        printt(f"Processing prompt: {prompt}\nMax length: {max_length}", color="y")

        # Generate text for each model
        generated_text_small = generate_text(model_small, tokenizer_small, prompt, max_length)
        generated_text_medium = generate_text(model_medium, tokenizer_medium, prompt, max_length)
        generated_text_large = generate_text(model_large, tokenizer_large, prompt, max_length)

        # Append results to output data
        output_data.append({
            "prompt": prompt,
            "max_length": max_length,
            "generated_text_small": generated_text_small,
            "generated_text_medium": generated_text_medium,
            "generated_text_large": generated_text_large
        })

    # Save the results in an output JSON file
    with open('experiment_results.json', 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

if __name__ == "__main__":
    main()
