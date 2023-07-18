import training_dataset as traind
import testing_dataset as testd
from transformers import ViTFeatureExtractor
import torch
import numpy as np
from datasets import load_metric


train_dataset = []
labels=[]
n_trainlabels=0


train_dataset,labels,n_trainlabels = traind.read_image()

test_dataset = []
labels=[]
n_testlabels=0
<<<<<<< HEAD
n_testsamples = {}

test_dataset,labels,n_testlabels,n_testsamples = testd.create_testing_dataset()

my_tensor = torch.tensor(my_list)
print(my_tensor)
print(type(my_tensor))

=======
test_dataset,labels,n_testlabels = testd.read_image()
>>>>>>> 1cd3e851c1c431f90c929f509da1b6d5435fe3d3
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
prepared_train = train_dataset_tensor.with_transform(preprocess)
# ... and the testing dataset
prepared_test = test_dataset_tensor.with_transform(preprocess)

def collate_fn(batch):
    return {'pixel_values': torch.stack([x['pixel_values'] for x in batch]),'labels': torch.tensor([x['label'] for x in batch])}


# accuracy metric
metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1),references=p.label_ids)
