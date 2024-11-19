# Multimodal Deep Generative Models

## Overview

This project focuses on developing and experimenting with multimodal deep generative models, particularly Variational Autoencoders (VAEs), that can handle data from multiple modalities. Currently, the model aggregates modalities using concatenation. The project includes:

- **Data visualization and manipulation** notebooks to help users get familiar with the datasets and data loaders.
- **Model manipulation** notebooks to define and experiment with multimodal deep generative models.
- **VAE Training** notebooks to train VAEs on individual modalities. After setting up the dataset directory, users can select the desired dataset, model, and training parameters directly within the notebook. Running the notebook trains the model and saves the resulting checkpoints and logs in the ./runs directory.

## Project Structure

    ├── data/                          # Directory for datasets (created during setup)
    ├── src/
    │   ├── datasets_manipulation.ipynb  # Notebook for data visualization and manipulation
    │   ├── models_manipulation.ipynb    # Notebook for model definition and experimentation
    │   ├── Train_VAE.ipynb             # Notebook for VAE training
    │   ├── models                       # Encoders, decoders and other model classes 
    │   └── dataset_manipulation         # Dataloaders related scripts                
    └──README.md                       # Project documentation and setup instructions



## Setup Instructions

### 1. Set Up the Data Directory and Download Datasets
Create a data directory and download the necessary datasets:

```bash
mkdir data
cd data

# Download the PolyMNIST dataset
curl -L -o data_ICLR_2.zip https://polybox.ethz.ch/index.php/s/wmAXzDAKn3Qogp7/download
unzip data_ICLR_2.zip 

# Download the CUB dataset
curl -L -o cub.zip http://www.robots.ox.ac.uk/~yshi/mmdgm/datasets/cub.zip
unzip cub.zip
```


### 2. Requirements

- numpy
- matplotlib
- torch
- torchvision
- torchnet
- jupyter
- json

