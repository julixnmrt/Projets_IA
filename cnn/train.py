import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import CNN  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),            
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,  
    shuffle=True,
    num_workers=0,  
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,  
    shuffle=False,
    num_workers=0,
)


architectures = [
    [3, 48, 96, 128],  
    [3, 64, 128, 128],  
]

epochs        = 20   
warmup_epochs = 3
initial_lr    = 0.0001
max_lr        = 0.001

results = {}


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def evaluate(model, loader):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs      = model(images)
            _, predicted = torch.max(outputs, 1)
            total       += labels.size(0)
            correct     += (predicted == labels).sum().item()
    return 100.0 * correct / total


########################################
# TRAINING LOOP
########################################

for arch in architectures:
    print("\n====================================")
    print(f"Training CNN architecture: {arch}")
    print("====================================\n")

    model     = CNN(channels=arch).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )

    best_acc = 0.0
    arch_str = "_".join(map(str, arch))

    for epoch in range(epochs):

        # WARMUP : une fois par epoch, avant les batches
        if epoch < warmup_epochs:
            warmup_lr = initial_lr + (max_lr - initial_lr) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total

        # Scheduler uniquement après le warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        test_acc = evaluate(model, testloader)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"cnn_best_{arch_str}.pth")

        print(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"LR: {get_lr(optimizer):.6f} | "
            f"Loss: {running_loss/len(trainloader):.4f} | "
            f"Train: {train_acc:.2f}% | "
            f"Test: {test_acc:.2f}% | "
            f"Best: {best_acc:.2f}%"
        )

    print(f"\n Best Test Accuracy for {arch}: {best_acc:.2f}%\n")
    results[str(arch)] = best_acc


########################################
# RÉSULTATS FINAUX
########################################

print("\n========== FINAL RESULTS ==========")
best_arch = max(results, key=results.get)
for arch, acc in results.items():
    marker = " ← BEST" if arch == best_arch else ""
    print(f"Architecture {arch} -> {acc:.2f}%{marker}")