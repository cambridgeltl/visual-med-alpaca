# Fine-tune GIT on a custom dataset for image captioning

import pandas as pd
import json
from datasets import load_dataset 
from torch.utils.data import Dataset
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
import torch
from tqdm import tqdm

# path to the csv containing training data directories
train_data_csv = ""
# path to the folder containing the training data images
train_data_folder = ""
# path to the csv containing training data directories
validation_data_csv = ""
# path to the folder containing the training data images
validation_data_folder = ""
# save pretrained model to
output_dir = ""


df = pd.read_csv(train_data_csv)
captions = [{"file_name": df.iloc[i]["name"], 
             "text": df.iloc[i]["caption"].strip()} for i in range(len(df))]

# add metadata.jsonl file to this folder
with open(train_data_folder + "metadata.jsonl", 'w') as f:
    for item in captions:
        f.write(json.dumps(item) + "\n")


df_val = pd.read_csv(validation_data_csv)
captions = [{"file_name": df_val.iloc[i]["name"], 
             "text": df_val.iloc[i]["caption"].strip()} for i in range(len(df_val))]

# add metadata.jsonl file to this folder
with open(validation_data_folder + "metadata.jsonl", 'w') as f:
    for item in captions:
        f.write(json.dumps(item) + "\n")


dataset = load_dataset("imagefolder", data_dir=train_data_folder, split="train")
val_dataset = load_dataset("imagefolder", data_dir=validation_data_folder, split="train")


# We use `GitProcessor` to turn each (image, text) pair into the expected inputs. Basically, the text gets turned into `input_ids` and `attention_mask`, and the image gets turned into `pixel_values`.
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")

        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding


processor = AutoProcessor.from_pretrained("microsoft/git-base")
train_dataset = ImageCaptioningDataset(dataset, processor)
validation_dataset = ImageCaptioningDataset(val_dataset, processor)

# Next, we create a corresponding [PyTorch DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), which allows us to get batches of data from the dataset.
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
validation_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=4)

model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")


# Dummy forward pass
batch = next(iter(train_dataloader))
outputs = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["input_ids"])
print(outputs.loss)


# Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)


num_epochs = 30
train_loss_history = []

for epoch in range(num_epochs):
    print("Epoch:", epoch)
    avg_loss = 0
    with tqdm(total=len(train_dataloader)) as pbar:
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            loss = outputs.loss
            train_loss_history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}, Loss {loss:.4f}, Avg Loss {avg_loss:.4f}")
    with torch.no_grad():
        model.eval()
        validation_loss = 0
        for batch_idx, batch in enumerate(validation_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            loss = outputs.loss
            validation_loss += loss.item()
        validation_loss /= len(validation_dataloader)
        print(f"Epoch {epoch}, Validation Loss {validation_loss:.4f}")

model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

