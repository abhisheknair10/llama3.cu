from transformers import AutoTokenizer
import json
import os

os.system("clear")

class TrieNode:
    def __init__(self):
        self.child = {}
        self.token = None

    # Method to count the number of children for each TrieNode
    def count_children(self):
        return len(self.child)


# Load the tokenizer from JSON
with open('model_weights/tokenizer.json') as f:
    tokenizer = json.load(f)
    print(len(str(tokenizer)))

# Extract the vocabulary from the tokenizer
tokenizer = tokenizer["model"]["vocab"]


# Build the trie structure
root = TrieNode()
for k, v in tokenizer.items():
    cursor = root
    for char in k:
        if char not in cursor.child:
            cursor.child[char] = TrieNode()
        cursor = cursor.child[char]
    cursor.token = v

# Function to traverse the trie and print the number of children for each node
NODES = 1  # Start with 1 to count the root
def print_children_count(node, depth=0):
    global NODES
    print(f"Node at depth {depth} has {node.count_children()} children")
    for child_char, child_node in node.child.items():
        NODES += 1  # Increment for every child node
        print_children_count(child_node, depth + 1)

# Print the number of children for each node in the trie
print_children_count(root)
print(f"Total number of nodes: {NODES}")

# Input text to tokenize
original_text = "If you are reading the data from the Internet instead, the same techniques can generally be used with the response you get from your HTTP API (it will be a file-like object); however, it is heavily recommended to use the third-party Requests library instead, which includes built-in support for JSON requests."
text = original_text.replace(" ", "Ġ")

cursor = root
tokens = []
for i, char in enumerate(text):
    if char in cursor.child:
        cursor = cursor.child[char]
    else:
        if cursor.token is not None:
            tokens.append(cursor.token)
            cursor = root

            if char in cursor.child:
                cursor = cursor.child[char]
            else:
                print(f"Whoops, unrecognized character: '{char}' at position {i}")
        else:
            print(f"Whoops, unrecognized token ending at character '{char}'")
            cursor = root

if cursor.token is not None:
    tokens.append(cursor.token)

original_model_id = "model_weights/"
original_tokenizer = AutoTokenizer.from_pretrained(original_model_id)
original_benchmark = original_tokenizer(original_text, return_tensors="pt")
original_benchmark = original_benchmark.input_ids.tolist()[0]

print(tokens == original_benchmark[1:])
print(tokens)
print(original_benchmark[1:])
