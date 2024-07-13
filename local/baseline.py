import torch
import numpy as np
import pandas as pd
from typing import Dict
import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)


### Model Parameters
# we will use with Distil-BERT
language_model_name = "distilbert-base-uncased"

### Training Argurments

# this GPU should be enough for this task to handle 32 samples per batch
batch_size = 32

# optim
learning_rate = 1e-4
weight_decay = 0.001 # we could use e.g. 0.01 in case of very low and very high amount of data for regularization

# training
epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

sst2_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli",trust_remote_code=True)



### METRIC DEFINITION

# Metrics
def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")

   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels,average="weighted")["f1"]
   return {"accuracy": accuracy, "f1": f1}


# MODEL
## Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(language_model_name,
                                                                   ignore_mismatched_sizes=True,
                                                                   output_attentions=False, output_hidden_states=False,
                                                                   num_labels=3) # number of the classes to change to 3

tokenizer = AutoTokenizer.from_pretrained(language_model_name)

# padding with the most long sentence!
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # avoid to use can reduce the memory on GPU

#examples are batch!
def tokenize_function(examples):
    examples["label"] = [labels_mapping[label] for label in examples["label"]]
    return tokenizer(examples["premise"], examples["hypothesis"],padding = True, truncation=True)




# Tokenize the dataset ...
print("Tokenize the dataset ...")
labels_mapping = {"ENTAILMENT":0, "CONTRADICTION":1, "NEUTRAL":2 }
tokenized_datasets_sst2 = sst2_dataset.map(tokenize_function, batched=True)

print(tokenized_datasets_sst2["train"][0])





#MODEL TRAINING

training_args = TrainingArguments(
    output_dir="training_dir",                    # output directory [Mandatory]
    num_train_epochs=epochs,                      # total number of training epochs
    per_device_train_batch_size=batch_size,       # batch size per device during training
    warmup_steps=500,                             # number of warmup steps for learning rate scheduler
    weight_decay=weight_decay,                    # strength of weight decay
    save_strategy="no",                           # save the model
    learning_rate=learning_rate,                  # learning rate
    gradient_checkpointing = True                 # to reduce memory usage
    # fp16 = True                                 # to reduce more memory usage
)


trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_datasets_sst2["train"],
   eval_dataset=tokenized_datasets_sst2["validation"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)


trainer.train()

trainer.evaluate()