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
        print(examples["image"])
        return examples
    dataset = dataset.map(transforms, batched=True)
    return dataset
read_image()
