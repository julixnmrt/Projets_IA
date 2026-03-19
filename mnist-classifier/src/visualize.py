import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

from model import MNISTClassifier


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Images directory
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=256)


# Load model
model = MNISTClassifier().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()


########################################
# 1️⃣ SHOW SAMPLE IMAGES
########################################

def show_samples():

    images, labels = next(iter(test_loader))

    plt.figure(figsize=(10,4))

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")

    plt.suptitle("Sample MNIST digits")

    plt.savefig(os.path.join(IMAGE_DIR, "samples.png"))
    plt.close()


########################################
# 2️⃣ SHOW MISCLASSIFIED DIGITS
########################################

def show_errors():

    errors = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs,1)

            for i in range(len(images)):

                if predicted[i] != labels[i]:
                    errors.append((images[i].cpu(), predicted[i].cpu(), labels[i].cpu()))

            if len(errors) >= 10:
                break


    plt.figure(figsize=(10,4))

    for i in range(10):

        img, pred, true = errors[i]

        plt.subplot(2,5,i+1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"P:{pred} T:{true}")
        plt.axis("off")

    plt.suptitle("Misclassified digits")

    plt.savefig(os.path.join(IMAGE_DIR, "errors.png"))
    plt.close()


########################################
# 3️⃣ CONFUSION MATRIX
########################################

def plot_confusion_matrix():

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs,1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.savefig(os.path.join(IMAGE_DIR, "confusion_matrix.png"))
    plt.close()


########################################
# 4️⃣ VISUALIZE NEURON WEIGHTS
########################################

def visualize_weights():

    weights = model.model[1].weight.data.cpu()

    plt.figure(figsize=(10,6))

    for i in range(10):

        w = weights[i].reshape(28,28)

        plt.subplot(2,5,i+1)
        plt.imshow(w, cmap="gray")
        plt.title(f"Neuron {i}")
        plt.axis("off")

    plt.suptitle("First layer neuron weights")

    plt.savefig(os.path.join(IMAGE_DIR, "weights.png"))
    plt.close()


########################################
# 5️⃣ t-SNE EMBEDDINGS
########################################

def tsne_visualization():

    features = []
    labels_list = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            x = model.model[:-1](images)

            features.append(x.cpu())
            labels_list.append(labels)

            if len(features) > 20:
                break


    features = torch.cat(features).numpy()
    labels_list = torch.cat(labels_list).numpy()

    tsne = TSNE(n_components=2, perplexity=30)

    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
        embeddings[:,0],
        embeddings[:,1],
        c=labels_list,
        cmap="tab10",
        s=10
    )

    plt.colorbar(scatter)
    plt.title("t-SNE projection of MNIST features")

    plt.savefig(os.path.join(IMAGE_DIR, "tsne.png"))
    plt.close()


########################################
# RUN VISUALIZATIONS
########################################

if __name__ == "__main__":

    show_samples()
    show_errors()
    plot_confusion_matrix()
    visualize_weights()
    tsne_visualization()

    print("All visualizations saved in /images")