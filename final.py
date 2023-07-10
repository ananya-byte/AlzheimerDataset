import training_dataset as traind
import dataset_vaish as testd
from transformers import ViTFeatureExtractor
import torch
import numpy as np
from datasets import load_metric


train_dataset = []
labels=[]
n_trainlabels=0
n_trainsamples = {}

train_dataset,labels,n_trainlabels,n_trainsamples = traind.create_training_dataset()

test_dataset = []
labels=[]
n_testlabels=0
n_testsamples = {}

test_dataset,labels,n_testlabels,n_testsamples = testd.create_testing_dataset()



# import model
model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
print(feature_extractor)



# device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def preprocess(batch):
    # take a list of PIL images and turn them to pixel values
    inputs = feature_extractor(batch['img'],return_tensors='pt')
    # include the labels
    inputs['label'] = batch['label']
    return inputs

# transform the training dataset
prepared_train = dataset_train.with_transform(preprocess)
# ... and the testing dataset
prepared_test = dataset_test.with_transform(preprocess)

def collate_fn(batch):
    return {'pixel_values': torch.stack([x['pixel_values'] for x in batch]),'labels': torch.tensor([x['label'] for x in batch])}


# accuracy metric
metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1),references=p.label_ids)
