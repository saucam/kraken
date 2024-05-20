import re
from transformers import AutoTokenizer

def extract_separators(template):
    """
    Extracts separators used in the tokenization template.
    """
    pattern = r"\{\{\s*([^{}]+?)\s*\+ message\['content'\] \+ eos_token \s*\}\}"
    matches = re.findall(pattern, template)
    separators = [match.strip() for match in matches]
    return separators

def recover_messages(formatted_message, separators, eos_token):
    """
    Recovers the original messages from the formatted message string.
    """
    split_messages = formatted_message.split(eos_token)
    
    if split_messages and split_messages[-1].strip() == '':
        split_messages.pop()

    recovered_messages = []
    alternate_roles = ["user", "assistant"]
    
    for index, message_content in enumerate(split_messages):
        if index == 0:
            role = "system"
        else:
            role = alternate_roles[(index - 1) % 2]

        clean_content = message_content.strip()
        for separator in separators:
            clean_content = clean_content.replace(separator.strip("'"), '', 1).strip()

        recovered_messages.append({"role": role, "content": clean_content})

    return recovered_messages

def recover_chat_messages(tokenized_chat, tokenizer):
    """
    Given a tokenized_chat string and a tokenizer, returns the list of message dictionaries.
    """
    jinja_template = tokenizer.chat_template
    separators = extract_separators(jinja_template)
    eos_token = tokenizer.eos_token
    recovered_messages = recover_messages(tokenized_chat, separators, eos_token)
    return recovered_messages

# Example usage
if __name__ == "__main__":
    checkpoint = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
        {"role": "assistant", "content": "None ... Humans cannot do that"},
        {"role": "user", "content": "Isn't there any gigantic human that can do that?"},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False)
    
    recovered_messages = recover_chat_messages(tokenized_chat, tokenizer)
    print(recovered_messages)
