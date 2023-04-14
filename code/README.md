# Visual Med-Alpaca Code


Important Notice: 

This project is produced based on [LLaMA](https://github.com/facebookresearch/llama), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), [Deplot](https://huggingface.co/docs/transformers/main/model_doc/deplot), [BigBio](https://huggingface.co/bigbio), [ROCO](https://github.com/razorx89/roco-dataset), [GenerativeImage2Text](https://github.com/microsoft/GenerativeImage2Text), [GPT-3.5-Turbo](https://platform.openai.com/docs/guides/chat), and is for academic purpose only. 

We are currently working on the necessary ethical clearance from University of Cambridge to determine whether, when and how could we provide complete inference code as well as an online interactive demo. 

**Currently you should still be able to easily reproduce the system with the follow this instruction.** We apologize for the inconvenience.

## Installation

**Hardware Requirement:**

Store all the data and models: 100GB free space. 

Deploy the full system (8bit inference): 32GB RAM and 24GB GPU memory. 

Train Med-Alpaca Lora and/or Med-GIT: 32GB RAM and 24GB GPU memory.

Train Med-Alpaca: 4 NVIDIA A100-SXM4-80GB.

[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) needs to be installed to take advantage GPU devices. 

**Environment (optional):**

[Install conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) and create conda virtual environment:

```
conda create -n visual-med-alpaca python=3.9
conda activate visual-med-alpaca
```

Setup hugging face model cache dir: 

```
export TRANSFORMERS_CACHE=/your/dir/to/huggingface_cache
```

**Installation:**

```
git clone https://github.com/cambridgeltl/visual-med-alpaca.git
cd visual-med-alpaca/code
pip install -r requirements.txt
```

## Data

Except datasets provided in [data folder](https://github.com/cambridgeltl/visual-med-alpaca/tree/main/data), you may download ROCO image data with official code below (might be slow) or from [unofficial kaggle](https://www.kaggle.com/datasets/virajbagal/roco-dataset).

```
git clone https://github.com/razorx89/roco-dataset.git
cd roco-dataset
python scripts/fetch.py
```

ROCO Dataset could take up to 20GB space, and take up to 8GB space when zipped. 

More details about these data can be found [here](https://github.com/cambridgeltl/visual-med-alpaca/tree/main/data).

## Training

**Med-Alpaca Lora**

configure `data_path` in `finetune-med.sh` and then

```
cd med-alpaca-lora
bash finetune-med.sh
```

**Med-Alpaca**

configure `data_path` in `train.sh` and then

```
cd med-alpaca
bash train.sh
```

**Med-GIT**

configure `train_data_csv`, `train_data_folder`, `validation_data_csv`, `validation_data_folder`, in `Fine_tune_GIT_on_an_image_captioning_dataset.py`

```
cd med-git
python Fine_tune_GIT_on_an_image_captioning_dataset.py
```

For more information, refer to [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), [Transformer-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials).

## Connect all pieces together

Finally, all these pieces could be combined in a workflow illustrated in the following diagram

image

We have tested [Deplot](https://huggingface.co/docs/transformers/main/model_doc/deplot) and [Med-GIT](https://github.com/cambridgeltl/visual-med-alpaca/tree/main/code/med-git) as **Medical Visual Foundation Model**. We have tested [Med-Alpaca Lora](https://github.com/cambridgeltl/visual-med-alpaca/tree/main/code/med-alpaca-lora), [Med-Alpaca](https://github.com/cambridgeltl/visual-med-alpaca/tree/main/code/med-alpaca), and [GPT-3.5-Turbo](https://platform.openai.com/docs/guides/chat) as  **Medical Lanugage Model**.

You can use the `image_caption` output by any **Medical Visual Foundation Model** to prompt any **Medical Lanugage Model** with the following template.

```
prompt_input = f"Below is an instruction that describes a task, paired with an input that provides further context of an uploaded image. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Input:\n{image_caption}\n\n### Response:\n"
prompt_no_input = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"
```
