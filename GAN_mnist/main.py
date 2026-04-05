# 1️⃣ imports et dataset
import torch, matplotlib.pyplot as plt, numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 2️⃣ modèle
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(-1, 1, 28, 28)
        return x_hat, z

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Autoencoder(latent_dim=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3️⃣ entraînement rapide
epochs = 5
for epoch in range(epochs):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        outputs, _ = model(imgs)
        loss = criterion(outputs, imgs)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 4️⃣ fonction d'interpolation


def interpolate_save(img1, img2, steps=10, filename="interpolation.png"):
    with torch.no_grad():
        z1 = model.encoder(img1.to(device))
        z2 = model.encoder(img2.to(device))
        imgs = []
        for alpha in np.linspace(0, 1, steps):
            z = (1-alpha)*z1 + alpha*z2
            out = model.decoder(z).view(28,28).cpu().numpy() * 255  # convert to 0-255
            out_img = Image.fromarray(out.astype(np.uint8))
            imgs.append(out_img)

        # Créer une grande image en concaténant horizontalement
        total_width = 28 * steps
        combined = Image.new('L', (total_width, 28))  # 'L' = grayscale
        for i, im in enumerate(imgs):
            combined.paste(im, (i*28, 0))
        combined.save(filename)
        print(f"Interpolation saved as {filename}")

def save_reconstructions_grid(dataset, indices=[0,1,2,3,4], filename="reconstructions.png", scale=10):
    """
    dataset : MNIST dataset
    indices : liste des indices à reconstruire
    scale : facteur d'agrandissement de l'image pour lisibilité
    """
    with torch.no_grad():
        imgs = []
        labels = []
        for idx in indices:
            img, label = dataset[idx]
            labels.append(label)
            img_device = img.unsqueeze(0).to(device)
            recon, _ = model(img_device)
            
            # convertir reconstruction en numpy 0-255
            recon_np = (recon.squeeze().cpu().numpy() * 255).astype(np.uint8)
            recon_img = Image.fromarray(recon_np).resize((28*scale,28*scale), Image.NEAREST)
            
            # ajouter le label au-dessus
            draw = ImageDraw.Draw(recon_img)
            draw.text((5, 5), str(label), fill=255)
            
            imgs.append(recon_img)
        
        # concaténer horizontalement
        total_width = 28*scale*len(imgs)
        combined = Image.new('L', (total_width, 28*scale))
        for i, im in enumerate(imgs):
            combined.paste(im, (i*28*scale,0))
        
        combined.save(filename)
        print(f"Reconstructions saved as {filename}")


save_reconstructions_grid(train_dataset, indices=[0,1,2,3,4], filename="reconstructions.png")