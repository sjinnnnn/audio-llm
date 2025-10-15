import torch
import torch.nn as nn

from modeling_llama import LlamaForCausalLM
from tokenization_llama import LlamaTokenizer
from functools import partial
import json

from tqdm import tqdm

import os
import random
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import re
import jiwer

model   = LlamaForCausalLM.from_pretrained(
    'sjinxxx/audio-llm-llama-3-8b',
    trust_remote_code=True, 
    device_map="auto",
    dtype=torch.bfloat16)



tokenizer = LlamaTokenizer.from_pretrained('tokenizer.model', trust_remote_code=True)

tokenizer.eos_token_id = tokenizer.special_tokens['<|end_of_text|>']

audio_encoder_prompt = '<audio>{}</audio><|startoftranscript|><|{}|><|transcribe|><|{}|><|notimestamps|><|wo_itn|>'

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, ds, prompt):
        self.datas = open(ds).readlines()
        self.prompt = prompt
        self.language = 'ko'

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio_path = data['audio_filepath']
        labels = data['text']

        return {
            'input_text': self.prompt.format(audio_path, self.language, self.language,), 
            'labels': labels,
        }

test_dataset = AudioDataset(ds="AI_HUB/Ksponspeech/eval_clean.jsonl", prompt=audio_encoder_prompt)

def collate_fn(inputs, tokenizer):
    input_texts = [_['input_text'] for _ in inputs]
    labels = [_['labels'] for _ in inputs]
    audio_info = [tokenizer.process_audio(_['input_text']) for _ in inputs ]
    input_tokens = tokenizer(input_texts,
                             return_tensors='pt',
                             padding='longest',
                             audio_info= audio_info)
 

    return input_tokens.input_ids, input_tokens.attention_mask, audio_info, labels


test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )


tokenizer.pad_token_id = tokenizer.eos_token_id
eos_id = tokenizer.special_tokens['<|end_of_text|>']
tokenizer.eos_token_id = eos_id


labels_text = []
prediction_text=[]

for idx, (input_ids,attention_mask, audio_info, labels) in enumerate(tqdm(test_dataloader, desc="Generating predictions")):
    output_ids = model.generate(input_ids.cuda(), attention_mask=attention_mask.cuda(), 
                                max_new_tokens=256, pad_token_id=tokenizer.eos_token_id ,audio_info = audio_info,
                                eos_token_id=tokenizer.eos_token_id ,min_new_tokens=1,num_return_sequences=1,
                                use_cache=False,) 
    input_length = input_ids.shape[1]
    generated_ids = output_ids[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    labels_text.append(labels[0])
    prediction_text.append(generated_text)
 

def normalize_text(text):
    text = re.sub(r"[^가-힣0-9a-zA-Z]", "", text)
    return text

refs_norm = [normalize_text(x) for x in labels_text]
preds_norm = [normalize_text(x) for x in prediction_text]

cer_score_norm_x= jiwer.cer(labels_text, prediction_text)
cer_score_norm= jiwer.cer(refs_norm, preds_norm)


print(f"CER: {cer_score_norm_x:.4f}")
print(f"CER_nrom: {cer_score_norm:.4f}")


