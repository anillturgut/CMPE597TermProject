import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import json
import glob
from tqdm import tqdm
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

train_size = int(len(train_images) * 0.05)
train_images, train_labels = train_images[:train_size], train_labels[:train_size]
test_size = int(len(test_images) * 0.05)
test_images, test_labels = test_images[:test_size], test_labels[:test_size]
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

mim_mael16 = torch.hub.load("ml-jku/MIM-Refiner", "mae_refined_l16")

#device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

device = "cpu"

mim_mael16.to(device)

transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])


def compute_embeddings(images: list) -> list:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = []
    
    with torch.no_grad():
      for image in tqdm(images):
        image = transform_image(image)[:3].unsqueeze(0)
        embeddings = mim_mael16(image.to(device))

        all_embeddings.append(np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist())



    return all_embeddings

print("Embedding is started")
embeddings = compute_embeddings(train_images)
print("Embedding is Completed!")
with open("_mimcifar100_all_embeddings.json", "w") as f:
        f.write(json.dumps(embeddings))

test_embeddings = compute_embeddings(test_images)

with open("_mimcifar100_all_embeddings_test.json", "w") as f:
        f.write(json.dumps(test_embeddings))