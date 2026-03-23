import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

from model import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################
# DATASET
########################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
])

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset,batch_size=256)

classes = test_dataset.classes


########################################
# VISUAL FUNCTIONS
########################################

def show_samples(save_dir):

    images,labels = next(iter(test_loader))

    plt.figure(figsize=(10,4))

    for i in range(10):

        img = images[i].permute(1,2,0)

        plt.subplot(2,5,i+1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis("off")

    plt.suptitle("Sample CIFAR10 images")
    plt.savefig(os.path.join(save_dir,"samples.png"))
    plt.close()


def show_errors(model,save_dir):

    errors=[]

    with torch.no_grad():

        for images,labels in test_loader:

            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            _,predicted=torch.max(outputs,1)

            for i in range(len(images)):

                if predicted[i]!=labels[i]:

                    errors.append((images[i].cpu(),
                                   predicted[i].cpu(),
                                   labels[i].cpu()))

            if len(errors)>=10:
                break


    plt.figure(figsize=(10,4))

    for i in range(10):

        img,pred,true=errors[i]

        img=img.permute(1,2,0)

        plt.subplot(2,5,i+1)
        plt.imshow(img)
        plt.title(f"P:{classes[pred]} T:{classes[true]}")
        plt.axis("off")

    plt.suptitle("Misclassified images")

    plt.savefig(os.path.join(save_dir,"errors.png"))
    plt.close()


def plot_confusion_matrix(model,save_dir):

    all_preds=[]
    all_labels=[]

    with torch.no_grad():

        for images,labels in test_loader:

            images=images.to(device)

            outputs=model(images)
            _,predicted=torch.max(outputs,1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())


    cm=confusion_matrix(all_labels,all_preds)

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.savefig(os.path.join(save_dir,"confusion_matrix.png"))
    plt.close()


def tsne_visualization(model,save_dir):

    features=[]
    labels_list=[]

    with torch.no_grad():

        for images,labels in test_loader:

            images=images.to(device)

            x=model.features(images)
            x=torch.flatten(x,1)

            features.append(x.cpu())
            labels_list.append(labels)

            if len(features)*images.shape[0]>2000:
                break


    features=torch.cat(features).numpy()
    labels_list=torch.cat(labels_list).numpy()

    tsne=TSNE(n_components=2,perplexity=30,random_state=42)

    embeddings=tsne.fit_transform(features)

    plt.figure(figsize=(8,6))

    scatter=plt.scatter(
        embeddings[:,0],
        embeddings[:,1],
        c=labels_list,
        cmap="tab10",
        s=10
    )

    cbar=plt.colorbar(scatter)
    cbar.set_ticks(range(10))
    cbar.set_ticklabels(classes)

    plt.title("t-SNE projection of CNN features")

    plt.savefig(os.path.join(save_dir,"tsne.png"))
    plt.close()


########################################
# RUN FOR BOTH MODELS
########################################

models_config = {
    "cnn_best_3_64_128_128.pth":[3,64,128,128],
    "cnn_best_3_48_96_128.pth":[3,48,96,128]
}


for model_file,channels in models_config.items():

    print("Processing",model_file)

    save_dir = f"images/{model_file.replace('.pth','')}"
    os.makedirs(save_dir,exist_ok=True)

    model = CNN(channels=channels).to(device)
    model.load_state_dict(torch.load(model_file,map_location=device))
    model.eval()

    show_samples(save_dir)
    show_errors(model,save_dir)
    plot_confusion_matrix(model,save_dir)
    tsne_visualization(model,save_dir)


print("All visualizations saved.")