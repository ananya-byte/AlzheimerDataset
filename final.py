import training_dataset as traind
import testing_dataset as testd
from transformers import ViTFeatureExtractor
import torch
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
from transformers import ViTForImageClassification
from transformers import Trainer
import pandas as pd
import matplotlib.pyplot as plt
import evaluate

train_dataset = traind.read_image()
test_dataset = testd.read_image()
# import model
model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
print(feature_extractor)



# device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def preprocess(batch):
    # take a list of PIL images and turn them to pixel values
    inputs = feature_extractor(batch['image'],return_tensors='pt')
    # include the labels
    inputs['label'] = batch['label']
    return inputs

# transform the training dataset
prepared_train = train_dataset.with_transform(preprocess)
# ... and the testing dataset
prepared_test = test_dataset.with_transform(preprocess)

def collate_fn(batch):
    return {'pixel_values': torch.stack([x['pixel_values'] for x in batch]),'labels': torch.tensor([x['label'] for x in batch])}


# accuracy metric
cfm_metric = evaluate.load("BucketHeadP65/confusion_matrix")
n= 0
metric = load_metric("accuracy")
def compute_metrics(p):
    predictions=np.argmax(p.predictions, axis=1)
    references=p.label_ids
    global n
    n = n+1
    if (n==11):
        results = cfm_metric.compute(references, predictions)
        print(results)
    return metric.compute(predictions,references)

training_args = TrainingArguments(
  output_dir="./cifar",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=4,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
)

labels = train_dataset.features['label'].names
model = ViTForImageClassification.from_pretrained(
    model_id,  # classification head
    num_labels=len(labels)
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_train,
    eval_dataset=prepared_test,
    tokenizer=feature_extractor,
)
train_results = trainer.train()
# save tokenizer with the model
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
# save the trainer state
trainer.save_state()


df = pd.DataFrame(trainer.state.log_history)
df.plot(kind='line')
# set the title
plt.title('LinePlots')
# show the plot
plt.show()
