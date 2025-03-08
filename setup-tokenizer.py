import json

# Load the JSON file as a string
with open('model_weights/tokenizer.json', 'r') as f:
    tokenizer_str = f.read()

# Replace 'Ġ' (represented as \u0120) with a space in the JSON string
modified_tokenizer_str = tokenizer_str.replace("\u0120", " ").replace(r"\\n", r"\n").replace("\u010A", "\n")

# Save the modified string back to the file
with open('model_weights/modified_tokenizer.json', 'w') as f:
    f.write(modified_tokenizer_str)

print("Modified JSON has been saved.")
