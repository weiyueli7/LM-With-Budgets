import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

sys.path.append("..")
from train.utils import *

def generate_text(model, tokenizer, prompt, max_length):
    """
    Generate text using the pretrained model
    :param model: Pretrained model
    :param tokenizer: Tokenizer
    :param prompt: Input prompt for the model
    :param max_length: Maximum length of the generated text
    :return: Generated text
    """
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt"
        )
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=1
        )
    return tokenizer.decode(
        output[0], 
        skip_special_tokens=True
        )

def main():
    parser = argparse.ArgumentParser(description="Generate text using a pretrained language model.")
    
    # Adding arguments
    parser.add_argument("--model_dir", type=str, default="TODO", help="Directory of the pretrained model")
    parser.add_argument("--prompt", type=str, default="What should we do for our quarter 2 project?", help="Input prompt for the model")
    parser.add_argument("--max_length", type=int, default=45, help="Maximum length of the generated text")

    args = parser.parse_args()

    printt(f"Generating text using pretrained model from {args.model_dir}\n----------", color="y")

    printt(f"Prompt: {args.prompt}\n----------", color="b")

    printt(f"Max length: {args.max_length}\n----------\n", color="r")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    # Generate and print the text
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_length)
    printt(f"Generated text:\n\n\n{generated_text}", 'g')

if __name__ == "__main__":
    main()
