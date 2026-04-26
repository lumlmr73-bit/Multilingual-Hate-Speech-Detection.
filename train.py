import os
import torch
import logging
from datasets import load_dataset
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

class HateSpeechDetector:
    def __init__(self, model_ckpt="xlm-roberta-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_ckpt)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(
            model_ckpt, 
            num_labels=2
        ).to(self.device)

    def preprocess_function(self, examples):
        return self.tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=128
        )

    def prepare_trainer(self):
        dataset = load_dataset("hate_eval", "multilingual", split='train[:1000]')
        tokenized_ds = dataset.map(self.preprocess_function, batched=True)
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=10
        )

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

if __name__ == "__main__":
    detector = HateSpeechDetector()
    trainer = detector.prepare_trainer()
    print("Model initialized successfully.")
