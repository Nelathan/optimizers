import random
import re
import ftfy
from datasets import load_dataset


def clean_sloppy_encoding(text):
    # use ftfy
    text = ftfy.fix_text(text)

    # unify newline style
    text = text.replace("\r", "\n")

    # replace tabs
    text = text.replace("\t", "    ")

    # replace fancy double quotes
    text = re.sub(r"[”“]", '"', text)

    # replace fancy single quotes
    text = re.sub(r"[‘’]", "'", text)

    # replace double single quotes with double quotes
    text = text.replace("''", '"')

    # replace fancy ellipses
    text = text.replace("…", "...")

    # replace expanded ellipses
    text = text.replace(". . .", "...")

    # remove whitespace preceding a comma or bang
    text = re.sub(r" +([,!])", r"\1", text)

    # remove trailing whitespace
    text = re.sub(r"([^ ]) +$", r"\1", text, flags=re.MULTILINE)

    # remove initial empty lines
    text = re.sub(r"^\n+", "", text)

    # remove other empty lines
    text = re.sub(r"\n\n+", "\n\n", text)

    # replace dinkus with spaced dinkus
    text = text.replace("***", "* * *")

    return text


def get_sugarquill(tokenizer, max_seq_length=8192):
    user_prompts = [
        "Tell me a story.",
        "I'd like to read something you've written.",
        "Write a story based on your profile.",
        "Please generate a creative narrative.",
        "I'd really enjoy reading a story from you.",
        "Stories from you always brighten my day. Please share one.",
        "If you have a story in mind, I'd love to hear it.",
        "Whenever you feel inspired, a story would be wonderful.",
        "Your storytelling always inspires me. Would you write one for me?",
        "I appreciate your creativity, share a story when you can.",
        "A story from you would be a treat. I'm excited to read it!",
        "I'm in the mood for something imaginative, if you're up for it.",
        "Your stories always make me think. I'd love to hear more.",
        "If you're feeling creative, I'd enjoy a story. I'm all ears!",
        "I always look forward to your stories.",
        "Whenever you're ready, I'd love to read something new. No pressure!",
        "Your stories are always a highlight for me.",
        "I'm always amazed by your creativity. Share a story when you can.",
        "Your unique perspective is refreshing. I'd love to hear a story that reflects it.",
    ]

    sugarquill = load_dataset("Nelathan/synthetic-sugar-quill", split="train[:10%]")

    sugarquill = sugarquill.map(
        lambda batch: {"text": [clean_sloppy_encoding(t) for t in batch["text"]]},
        batched=True,
        desc="Cleaning text",
    )

    sugarquill = sugarquill.map(
        lambda batch: {
            "conversations": [
                [
                    {
                        "role": "system",
                        "content": f"You are a creative writing AI model. Your author profile and personality are described below. Adhere to this profile when writing.\n\n# Author Profile\n{profile}",
                    },
                    {
                        "role": "user",
                        "content": random.choice(user_prompts),
                    },
                    {
                        "role": "assistant",
                        "content": text,
                    },
                ]
                for profile, text in zip(batch["profile"], batch["text"])
            ],
        },
        batched=True,
        remove_columns=sugarquill.column_names,
        desc="Assembling Sugarquill",
    )

    def tokenize_and_mask(batch):
        return tokenizer.apply_chat_template(
            batch["conversations"],
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            tokenize=True,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_dict=True,
        )

    sugarquill = sugarquill.map(
        tokenize_and_mask,
        batched=True,
        remove_columns=sugarquill.column_names,
        desc="Tokenizing Sugarquill",
        num_proc=1,
        batch_size=64,
    )

    return sugarquill
