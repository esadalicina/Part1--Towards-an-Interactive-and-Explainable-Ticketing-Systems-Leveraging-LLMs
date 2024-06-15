from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_title(text, max_length=10):
    prompt = f"Generate a short title for the following text: {text}"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return title

text = "Your long text goes here."
title = generate_title(text)
print("Generated Title:", title)

# Save the model and tokenizer
model.save_pretrained("/home/users/elicina/Master-Thesis/Models/TitleGen")
tokenizer.save_pretrained("/home/users/elicina/Master-Thesis/Models/TokTitleGen")

print("Model and tokenizer saved successfully.")