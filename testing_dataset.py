from pathlib import Path
import os
from datasets import load_dataset


# Dataset Parameters - CHANGE HERE
DATASET_PATH = 'dataset\\test' # the dataset file or root folder path.

# Image Parameters
n_classes = 4 # CHANGE HERE, total number of classes
img_height = 224# CHANGE HERE, the image height to be resized to
img_width = 224 # CHANGE HERE, the image width to be resized to
channels = 3 
batch_size= 130

def read_image():
    dataset = load_dataset("imagefolder", data_dir=DATASET_PATH,drop_labels=False,split="train")             
    def transforms(examples):
        examples["image"] = [image.convert("RGB").resize((224,224)) for image in examples["image"]]
        return examples
    dataset = dataset.map(transforms, batched=True)
<<<<<<< HEAD
    return dataset
=======
    labels = dataset["train"].features['label'].names
    num_classes = len(labels)
<<<<<<< HEAD
    print(labels)
    print(dataset["train"][0]['image'].resize((200,200)))
    return dataset,labels,num_classes
    
    
read_image()
=======
    return dataset,labels,num_classes
>>>>>>> 1cd3e851c1c431f90c929f509da1b6d5435fe3d3
>>>>>>> 9d653f0cb85416c7afc430e5902ccda07076ca9e
