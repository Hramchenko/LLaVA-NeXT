# %%
import sys
sys.path.append(".")

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image

conversation_lib.default_conversation = conversation_lib.conv_templates["llava_llama_3"]

# %%
conversation_lib.default_conversation.tokenizer

# %%
conv = conversation_lib.default_conversation.copy()
conv.tokenizer = conversation_lib.default_conversation.tokenizer

# %%
full_conversation = conv.get_prompt(add_generation_prompt=False)

# %%
full_conversation

# %%
assistant_begin = "<|start_header_id|>assistant<|end_header_id|>\n\n"
assistant_end = "<|eot_id|>"

# %%


# %%
full_conversation.find(assistant_begin, 383+1)

# %%


# %%


# %%


# %%
conversation_lib.default_conversation.get_prompt()
conversation_lib.default_conversation.tokenizer.pad_token = conversation_lib.default_conversation.tokenizer.eos_token

# %%
tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/hdd_0/user/ML/LLM/llama3-llava-next-8b-tokenizer/",
            #"/hdd_0/user/ML/LLM/llama3-llava-next-8b/",
            
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
tokenizer.pad_token = tokenizer.eos_token

# %%
def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    conv.tokenizer = conversation_lib.default_conversation.tokenizer
    #conv.tokenizer = tokenizer
    #conv.sep_style = conversation_lib.SeparatorStyle.LLAMA_3
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt(add_generation_prompt=False))

    # Tokenize conversations

    print("conversations", conversations)
    #print("yyyy", conv.sep_style)

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    assistant_begin = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    assistant_end = "<|eot_id|>"
    if has_image:
        print("HAS IMAGE")
        #raise Exception()
    for conversation, target in zip(conversations, targets):
        current_index = 0
        current_token = 0
        print("conv:", conversation)
        print("TGT:", target)
        while True:
            assistant_begin_idx = conversation.find(assistant_begin, current_index)
            if assistant_begin_idx < 0:
                break
            
            
            a_start = assistant_begin_idx + len(assistant_begin)
            start_text = conversation[current_index:a_start]
            if has_image:
                 start_tokens_cnt = len(tokenizer_image_token(start_text, tokenizer))
            else:
                start_tokens_cnt = len(tokenizer(start_text).input_ids)
            print(f"start_text '{start_text}'")
            print(f"start: {tokenizer(start_text).input_ids}")
            print("IGNORE", current_token, current_token + start_tokens_cnt)
            print("IGN",  target[current_token: current_token + start_tokens_cnt])

            target[current_token: current_token + start_tokens_cnt] = IGNORE_INDEX
            print(target.shape, current_index, a_start)
            a_end = conversation.find(assistant_end, a_start) + len(assistant_end)
            current_index = a_end
            answer_text = conversation[a_start: a_end]
            print(f"answer text '{answer_text}'")
            print(f"answer: {tokenizer(answer_text).input_ids}")
            instr_tokens_cnt = len(tokenizer(answer_text).input_ids)
            current_token += start_tokens_cnt + instr_tokens_cnt
    print("SIZE", target.shape, current_token)



    # # Mask targets
    # #sep = conv.sep + conv.roles[1]
    # sep = "<|start_header_id|>assistant<|end_header_id|>\n\n" #conv.roles[1]
    # print("sep", sep)
    # for conversation, target in zip(conversations, targets):
    #     total_len = int(target.ne(tokenizer.pad_token_id).sum())

    #     rounds = conversation.split(sep)
    #     print("rounds", rounds)
    #     re_rounds = [sep.join(rounds[:3])]
    #     #print("uuu", re_rounds)
    #     for conv_idx in range(3, len(rounds), 2):
    #         re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))
    #     cur_len = 0
    #     target[:cur_len] = IGNORE_INDEX
    #     for i, rou in enumerate(re_rounds):
    #         if rou == "":
    #             break

    #         parts = rou.split(sep)
    #         if len(parts) != 2:
    #             break
    #         parts[0] += sep

    #         if has_image:
    #             round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
    #             instruction_len = len(
    #                 tokenizer_image_token(parts[0], tokenizer))
    #         else:
    #             round_len = len(tokenizer(rou).input_ids) + 1
    #             instruction_len = len(tokenizer(parts[0]).input_ids)

    #         if i > 0:
    #             round_len -= 1
    #             instruction_len -= 1

    #         target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

    #         cur_len += round_len
    #     target[cur_len:] = IGNORE_INDEX

    #     if cur_len < tokenizer.model_max_length:
    #         if cur_len != total_len:
    #             #target[:] = IGNORE_INDEX
    #             print(
    #                 f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
    #                 f" (ignored)"
    #             )
    print("RESULT", targets)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

# %%
preprocess_llama3([[{'from': 'human', 'value': '<image>\nWhat kind of animal is this?'}, {'from': 'gpt', 'value': 'dog'}, {'from': 'human', 'value': 'Is it big?'}, {'from': 'gpt', 'value': 'no. its little'}]], tokenizer=tokenizer)


print('***********************************************************')
preprocess_llama3([[{'from': 'human', 'value': '<image>\nWhat kind of animal is this?'}, {'from': 'gpt', 'value': 'dog'}]], tokenizer=tokenizer)

print('*********************************')
preprocess_llama3([[{'from': 'human', 'value': '<image>\nWhat kind of animal is this?'}, {'from': 'gpt', 'value': 'cat'}]], tokenizer=tokenizer, has_image=True)

# %%
