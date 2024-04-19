from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import torch
import gc


def evaluation_llm(file_path=None, string=None):
    tokenizer = AutoTokenizer.from_pretrained('/home/rijkaa/leraa/solution/deberta-xsmall-finetuned/checkpoint-14484')
    model = AutoModelForSequenceClassification.from_pretrained('/home/rijkaa/leraa/solution/deberta-xsmall-finetuned/checkpoint-14484/')
    trainer_args = TrainingArguments(
        'tmp_trainer',
        per_device_eval_batch_size=4,
    )
    trainer = Trainer(
        model,
        trainer_args,
        tokenizer=tokenizer
    )
    def preprocess_function(example):
        return tokenizer(example['text'], max_length=128, padding=True, truncation=True)
    if file_path is not None:
        with open(file_path) as file:
            string = file.read()
    
    if string is not None:
        data = Dataset.from_dict({'text':[string]})
        tokenized_data = data.map(preprocess_function, batched=True)
        logits = trainer.predict(tokenized_data).predictions
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        # print(probs[0, 1])
        if probs[0, 1] > 0.5:
            print('Generated')
        else:
            print('Natural')
    
    torch.cuda.empty_cache()
    gc.collect()