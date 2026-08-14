import copy
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    """
    Task 1 — Building and Training a CNN from Scratch on MNIST

    Seven CNN architectures trained and compared on MNIST.
    Final optimal model trained with data augmentation.

    Run:
        python task1_cnn_from_scratch_on_mnist.py

    Output files saved to working directory:
        mnist_samples.png              sample images per class
        all_experiments_curves.png     training curves for all 7 models
        optimal_curves.png             optimal model training curve
        optimal_confusion_matrix.png   confusion matrix on test set
        misclassified.png              20 misclassified examples
        optimal_cnn_mnist.pth          saved model weights

    Author: Tamilore Fakeye
    """



    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')



    # convert to tensor and normalise with MNIST mean/std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load datasets
    full_train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Split training into train (55,000) and validation (5,000)
    train_size = 55000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoaders
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'Train samples : {len(train_dataset)}')
    print(f'Val   samples : {len(val_dataset)}')
    print(f'Test  samples : {len(test_dataset)}')

    # Display a sample of images from each class
    classes = [str(i) for i in range(10)]

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('MNIST Dataset – Sample Images per Class', fontsize=14)

    # Collect one image per class
    seen = {}
    for img, label in full_train_dataset:
        if label not in seen:
            seen[label] = img
        if len(seen) == 10:
            break

    for i, ax in enumerate(axes.flatten()):
        img = seen[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Class: {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.show()



    def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, name='Model'):
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        best_weights = copy.deepcopy(model.state_dict())

        for epoch in range(epochs):
            # Training
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc  = 100.0 * correct / total

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total   += labels.size(0)

            val_loss = val_loss / val_total
            val_acc  = 100.0 * val_correct / val_total

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            scheduler.step()

            print(f'[{name}] Epoch {epoch+1:02d}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')

        print(f'Best Val Acc: {best_val_acc:.2f}%')
        model.load_state_dict(best_weights)
        return history, model


    def evaluate_on_test(model, test_loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))
        return np.array(all_labels), np.array(all_preds)


    def plot_history(histories, names, save_name='curves.png'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

        for history, name, color in zip(histories, names, colors):
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], '--', color=color, alpha=0.6)
            axes[0].plot(epochs, history['val_loss'],   '-',  color=color, label=name)
            axes[1].plot(epochs, history['train_acc'],  '--', color=color, alpha=0.6)
            axes[1].plot(epochs, history['val_acc'],    '-',  color=color, label=name)

        axes[0].set_title('Loss (solid=val, dashed=train)')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].set_title('Accuracy (solid=val, dashed=train)')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.show()


    def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_name='cm.png'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.title(title)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.show()


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



    class BaselineCNN(nn.Module):
        def __init__(self):
            super(BaselineCNN, self).__init__()
            # Block 1: 1 -> 32 feature maps, 3x3 kernel
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)

            # Block 2: 32 -> 64 feature maps
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)

            # Classifier
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    baseline_model = BaselineCNN()
    print(f'Baseline CNN parameters: {count_parameters(baseline_model):,}')
    print(baseline_model)

    EPOCHS = 10
    history_baseline, model_baseline = train_model(
        BaselineCNN(), train_loader, val_loader, epochs=EPOCHS, name='Baseline'
    )



    class DeepCNN(nn.Module):
        def __init__(self):
            super(DeepCNN, self).__init__()
            self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)   # 28 -> 14

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)   # 14 -> 7

            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            # No pool after conv3 to preserve spatial resolution at 7x7

            self.fc1 = nn.Linear(128 * 7 * 7, 256)
            self.fc2 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    print(f'Deep CNN parameters: {count_parameters(DeepCNN()):,}')
    history_deep, model_deep = train_model(
        DeepCNN(), train_loader, val_loader, epochs=EPOCHS, name='DeepCNN'
    )



    class CNNWithActivation(nn.Module):
        def __init__(self, activation='relu'):
            super(CNNWithActivation, self).__init__()
            activations = {
                'relu':      nn.ReLU(),
                'leakyrelu': nn.LeakyReLU(0.1),
                'elu':       nn.ELU()
            }
            self.act = activations[activation]

            self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

            self.fc1 = nn.Linear(128 * 7 * 7, 256)
            self.fc2 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool1(self.act(self.conv1(x)))
            x = self.pool2(self.act(self.conv2(x)))
            x = self.act(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = self.act(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    history_leaky, model_leaky = train_model(
        CNNWithActivation('leakyrelu'), train_loader, val_loader, epochs=EPOCHS, name='LeakyReLU'
    )
    history_elu, model_elu = train_model(
        CNNWithActivation('elu'), train_loader, val_loader, epochs=EPOCHS, name='ELU'
    )



    class CNNWithBatchNorm(nn.Module):
        def __init__(self):
            super(CNNWithBatchNorm, self).__init__()
            self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
            self.bn1   = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)

            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3   = nn.BatchNorm2d(128)

            self.fc1     = nn.Linear(128 * 7 * 7, 256)
            self.fc2     = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.4)

        def forward(self, x):
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    print(f'BN CNN parameters: {count_parameters(CNNWithBatchNorm()):,}')
    history_bn, model_bn = train_model(
        CNNWithBatchNorm(), train_loader, val_loader, epochs=EPOCHS, name='BatchNorm'
    )



    class CNNAvgPool(nn.Module):
        def __init__(self):
            super(CNNAvgPool, self).__init__()
            self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
            self.bn1   = nn.BatchNorm2d(32)
            self.pool1 = nn.AvgPool2d(2, 2)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm2d(64)
            self.pool2 = nn.AvgPool2d(2, 2)

            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3   = nn.BatchNorm2d(128)

            self.fc1     = nn.Linear(128 * 7 * 7, 256)
            self.fc2     = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.4)

        def forward(self, x):
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    history_avg, model_avg = train_model(
        CNNAvgPool(), train_loader, val_loader, epochs=EPOCHS, name='AvgPool'
    )



    class CNNLargeKernel(nn.Module):
        def __init__(self):
            super(CNNLargeKernel, self).__init__()
            self.conv1 = nn.Conv2d(1,  32, kernel_size=5, padding=2)  # 5x5 kernel
            self.bn1   = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)

            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3   = nn.BatchNorm2d(128)

            self.fc1     = nn.Linear(128 * 7 * 7, 256)
            self.fc2     = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.4)

        def forward(self, x):
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    history_lk, model_lk = train_model(
        CNNLargeKernel(), train_loader, val_loader, epochs=EPOCHS, name='5x5Kernel'
    )



    all_histories = [history_baseline, history_deep, history_leaky, history_elu,
                     history_bn, history_avg, history_lk]
    all_names     = ['Baseline', 'DeepCNN', 'LeakyReLU', 'ELU',
                     'BatchNorm', 'AvgPool', '5x5Kernel']

    plot_history(all_histories, all_names, save_name='all_experiments_curves.png')

    # Summary table
    print(f"{'Model':<15} {'Best Val Acc':>12} {'Final Train Acc':>15}")
    print('-' * 45)
    for name, history in zip(all_names, all_histories):
        best_val = max(history['val_acc'])
        final_train = history['train_acc'][-1]
        print(f"{name:<15} {best_val:>11.2f}% {final_train:>14.2f}%")



    class OptimalCNN(nn.Module):
        def __init__(self):
            super(OptimalCNN, self).__init__()
            # Block 1
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),   # -> 14x14
                nn.Dropout2d(0.25)
            )
            # Block 2
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),   # -> 7x7
                nn.Dropout2d(0.25)
            )
            # Classifier head
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    transform_augmented = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_augmented = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=transform_augmented
    )
    train_aug, val_aug = random_split(
        full_train_augmented, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader_aug = DataLoader(train_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f'Optimal CNN parameters: {count_parameters(OptimalCNN()):,}')
    history_optimal, model_optimal = train_model(
        OptimalCNN(), train_loader_aug, val_loader, epochs=20, lr=1e-3, name='OptimalCNN'
    )



    print('Optimal CNN – Test Set Evaluation')
    y_true, y_pred = evaluate_on_test(model_optimal, test_loader)

    test_acc = 100.0 * (y_true == y_pred).sum() / len(y_true)
    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')

    plot_confusion_matrix(y_true, y_pred,
                          title='Optimal CNN – Confusion Matrix (Test Set)',
                          save_name='optimal_confusion_matrix.png')

    plot_history([history_optimal], ['OptimalCNN'], save_name='optimal_curves.png')



    # Collect misclassified images
    model_optimal.eval()
    misclassified_imgs, misclassified_true, misclassified_pred = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_optimal(images.to(device))
            _, predicted = outputs.max(1)
            mask = predicted.cpu() != labels
            misclassified_imgs.extend(images[mask])
            misclassified_true.extend(labels[mask].tolist())
            misclassified_pred.extend(predicted.cpu()[mask].tolist())
            if len(misclassified_imgs) >= 20:
                break

    fig, axes = plt.subplots(4, 5, figsize=(12, 9))
    fig.suptitle('Misclassified Examples – Optimal CNN', fontsize=13)
    for i, ax in enumerate(axes.flatten()):
        if i >= len(misclassified_imgs): break
        img = misclassified_imgs[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True:{misclassified_true[i]} Pred:{misclassified_pred[i]}', fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Total misclassified on test set: {(y_true != y_pred).sum()} / {len(y_true)}')



    torch.save(model_optimal.state_dict(), 'optimal_cnn_mnist.pth')
    print('Model saved to optimal_cnn_mnist.pth')
