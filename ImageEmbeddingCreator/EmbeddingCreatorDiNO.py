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

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

dinov2_vits14.to(device)

transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])


def compute_embeddings(images: list) -> list:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = []
    
    with torch.no_grad():
      for image in tqdm(images):
        image = transform_image(image)[:3].unsqueeze(0)
        embeddings = dinov2_vits14(image.to(device))

        all_embeddings.append(np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist())



    return all_embeddings


embeddings = compute_embeddings(train_images)

with open("_dinocifar100_all_embeddings.json", "w") as f:
       f.write(json.dumps(embeddings))

test_embeddings = compute_embeddings(test_images)

with open("_dinocifar100_all_embeddings_test.json", "w") as f:
        f.write(json.dumps(test_embeddings))