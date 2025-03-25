# Leveraging Sparse Annotations for Leukemia Diagnosis on the Large Leukemia Dataset

![architecture_AttriDet](https://github.com/intelligentMachines-ITU/SLA-Det_large_leukemia_dataset/blob/main/SLA_Det_archi2-1.png)


**Authors:** Abdul Rehman, Talha Meraj, Aiman Mahmood Minhas, Ayisha Imran, Mohsen Ali, Waqas Sultani, Mubarak Shah

**This is the journal version of "A Large-scale Multi Domain Leukemia Dataset for the White Blood Cells Detection with Morphological Attributes for Explainability (MICCAI-2024)"**
**MIA 2025**
**Paper:** [ArXiv](https://arxiv.org/abs/)

**Abstract:** _ELeukemia is 10th most frequently diagnosed cancer and one of the leading causes of cancer-related deaths worldwide. Realistic analysis of Leukemia requires White Blook Cells (WBC) localization, classification, and morphological assessment. Despite deep learning advances in medical imaging, leukemia analysis lacks a large, diverse multi-task dataset, while existing small datasets lack domain diversity, limiting real-world applicability. To overcome dataset
challenges, we present a large-scale WBC dataset named ‘Large Leukemia Dataset’ (LLD) and novel methods for detecting WBC with their attributes. Our contribution here is threefold. First, we present a large-scale Leukemia dataset collected through Peripheral Blood Films (PBF) from several patients, through multiple microscopes, multi-cameras, and multi-magnification. To enhance diagnosis explainability and medical expert acceptance, each leukemia cell is annotated at 100x with 7 morphological attributes, ranging from Cell Size to Nuclear Shape. Secondly, we propose a multi-task model that not only detects WBCs but also predicts their attributes, providing an interpretable and clinically meaningful solution. Third, we propose a method for WBC detection with attribute analysis using sparse annotations. This approach reduces the annotation burden on hematologists, requiring them to mark only a small area within the field of view. Our method enables the model to leverage the entire field of view rather than just the annotated regions, enhancing learning efficiency and diagnostic accuracy. From diagnosis explainability to overcoming domain-shift challenges, presented datasets could be used for many challenging aspects of microscopic image analysis._

# Installation

We recommend the use of a Linux machine equipped with CUDA-compatible GPUs. The execution environment can be installed through Conda.

Clone repo:
```
git clone https://github.com/intelligentMachines-ITU/LLD-Large-Leukemia-dataset-for-microscopic-imagery.git
cd LLD-Large-Leukemia-dataset-for-microscopic-imagery
```
 
Conda
Install requirements.txt in a Python>=3.7.16 environment, requiring PyTorch version 1.13.1 with CUDA version 11.7 support. The environment can be installed and activated with:
```
conda create --name SLA_Det python=3.7.16
conda activate SLA_Det
pip install -r requirements.txt  # install
```

# Dataset 
The sparse LeukemiaAttri dataset can be downloaded from the given link:

[Large Leukemia Dataset](https://drive.google.com/drive/folders/1VJSM5d1ndKtz4AQy7zQfnVGyiyGjRl8W?usp=sharing)


# JSON COCO Format
```
|-COCO Dataset
      |---Annotations
                     |---train.json
                     |---test.json
      |---Images
                |---train
                |---test
```

# YOLO Format

We construct the training and testing set for the yolo format settings, dataset can be downloaded from:

labels prepared in YOLO format but with attributes information as: cls x y w h px1 px2 py1 py2 a1 a2 a3 a4 a5 a6 whereas standard yolo format of labels was cls x y w h 

data -> WBC_v1.yaml
```
train: ../images/train
test: ../images/test


# number of classes
nc: 14

# class names
names: ["None","Myeloblast","Lymphoblast", "Neutrophil","Atypical lymphocyte","Promonocyte","Monoblast","Lymphocyte","Myelocyte","Abnormal promyelocyte", "Monocyte","Metamyelocyte","Eosinophil","Basophil"]
```

# Training
To reproduce the experimental result, we recommend training the model with the following steps.

Before training, please check data/WBC_v1.yaml, and enter the correct data paths.

The model is trained in 2 successive phases:

Phase 1: Model pre-train # 100 Epochs

Phase 2: Pre-trained weights used for further training # 30 Epochs


# Phase 1: Model pre-train
The first phase of training consists in the pre-training of the model. Training can be performed by running the following bash script:
Pre_trained weights can be downloaded from [here.]([[https://drive.google.com/drive/folders/1GTmefJJQyVaZ3qaCdfhvryWX9kNdKP80?usp=sharing](https://drive.google.com/drive/folders/1VJSM5d1ndKtz4AQy7zQfnVGyiyGjRl8W?usp=sharing)](https://drive.google.com/drive/folders/1Bg62RFVXwcoJP2VS3eqn1dIISwv1HHFo?usp=sharing))
```
python pre_train.py \
 --name AttriDet_Phase1 \
 --batch 8 \
 --imgsz 640 \
 --epochs 100 \
 --data data/WBC_pre.yaml \
 --hyp data/hyps/hyp.scratch-high.yaml
 --weights yolov5x.pt
```

# Phase 2: Pre-trained weights used for further training 
The Pre-trained weights used for further training. Training can be performed by running the following bash script:


```
python train.py \
 --name AttriDet_Phase2 \
 --batch 4 \
 --imgsz 640 \
 --epochs 30 \
 --data data/WBC_v1.yaml \
 --hyp data/hyps/hyp.scratch-high.yaml
 --weights runs/AttriDet_Phase1/weights/last.pt
```

# Testing phase
once model training will be done, an Attribute_model directory will be created, containing ground truth vs predicted attributes csv files, additionally it will contain the attribute model weights saved with f1 score as best weights whereas last.pt will also be saved. These files and weights will be saved based on validation of model. To get model testing, the last.pt of SLA_Detector and last.pt of attribute model will be used to run the test.py file. In result, in Attribute_model directory, a test subdirectory will be created, containing test.csv of ground truth vs predicted attributes. The yolo weights and testing will be save correspondingly in runs/val/exp.

```
python test.py \
 --weights /runs/train/SLA_Det/weights//last.pt,
 --data, data/WBC_v1.yaml, 
 --save-csv,
 --imgsz,640
```
