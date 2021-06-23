from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pandas as pd


raw_datasets = load_dataset('csv', data_files='datasets/train_funding.csv')['train'].train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets['train']
small_eval_dataset = tokenized_datasets['test']

# train_list = small_train_dataset['label']
# test_list = small_eval_dataset['label']

# small_train_dataset['label'] = [int(i) for i in train_list]
# small_eval_dataset['label'] = [int(i) for i in test_list]

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")

training_args = TrainingArguments("test_trainer", per_device_train_batch_size = 2)

trainer = Trainer(
    model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
)

trainer.train()

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
print(trainer.evaluate())