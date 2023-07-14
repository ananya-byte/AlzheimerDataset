import training_dataset as traind
import dataset_vaish as testd
from transformers import ViTFeatureExtractor

train_dataset = []
labels=[]
n_trainlabels=0
n_trainsamples = {}

train_dataset,labels,n_trainlabels,n_trainsamples = traind.create_training_dataset()
print (train_dataset) 
test_dataset = []
labels=[]
n_testlabels=0
n_testsamples = {}

test_dataset,labels,n_testlabels,n_testsamples = testd.create_testing_dataset()



# import model
model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
print(feature_extractor)

example = feature_extractor(train_dataset[0]['img'],return_tensors='pt')
print(example)


