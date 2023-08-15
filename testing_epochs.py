import training_dataset as traind
import testing_dataset as testd
import gui
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
from datasets import load_dataset, Image
import PySimpleGUI as sg
import os.path
from PIL import Image
import os
import time
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
    results = cfm_metric.compute(references=p.label_ids, predictions=np.argmax(p.predictions, axis=1))
    print(results)
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids
    )

training_args = TrainingArguments(
  output_dir="./cifar",
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=4,
  save_steps=300,
  eval_steps=300,
  logging_steps=300,
  learning_rate=5e-4,
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


# img_viewer.py


# First the window layout in 2 columns

file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Text(size=(40, 1), key="-RESULT-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)

layout = [
            [sg.Image(r'C:\PySimpleGUI\Logos\PySimpleGUI_Logo_320.png')],
         ]

    # Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
        # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0])
            window["-TOUT-"].update(values["-FILE LIST-"][0].split("_")[0])
            window["-IMAGE-"].update(filename=filename)
            dataset = Dataset.from_dict({"image": [filename]}).cast_column("image", Image())
            image = dataset["image"][0].resize((200,200))
            window["-TOUT-"].update("working1")
            inputs = feature_extractor(image, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                window["-TOUT-"].update("working3")
            predicted_label = logits.argmax(-1).item()
            labels = dataset.features['label']
            window["-TOUT-"].update("working")
        except:
            pass
window.close()
