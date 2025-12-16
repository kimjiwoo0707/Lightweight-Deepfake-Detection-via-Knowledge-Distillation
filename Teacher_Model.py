import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import time

CFG = {
    'train_dir': '/home/work/jiu/Deepfake/train',
    'test_dir': '/home/work/jiu/Deepfake/test',
    'val_dir': '/home/work/jiu/Deepfake/valid',
    'batch_size': 16,
    'img_size': 224,
    'resize_size': 256,
    'learning_rate': 1e-5,
    'epochs': 20,
    'patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_model_name': 'efficientnetb7_teacher.pth',
    'fc_hidden_dim': 512,
    'dropout': 0.3,
}

train_transforms = transforms.Compose([
    transforms.Resize(CFG['resize_size']),
    transforms.CenterCrop(CFG['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_val_transforms = transforms.Compose([
    transforms.Resize(CFG['resize_size']),
    transforms.CenterCrop(CFG['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=CFG['train_dir'], transform=train_transforms)
test_dataset = datasets.ImageFolder(root=CFG['test_dir'], transform=test_val_transforms)
val_dataset = datasets.ImageFolder(root=CFG['val_dir'], transform=test_val_transforms)

train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['batch_size'], shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], shuffle=False)

device = torch.device(CFG['device'])

class EfficientNetB7Custom(nn.Module):
    def __init__(self, pretrained=True, fc_hidden_dim=512, dropout=0.3):
        super().__init__()
        self.model = models.efficientnet_b7(pretrained=pretrained)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

efficient_model = EfficientNetB7Custom(
    pretrained=True,
    fc_hidden_dim=CFG['fc_hidden_dim'],
    dropout=CFG['dropout']
).to(device)

def fit_model(model, train_loader, val_loader, cfg):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg['epochs']):
        print(f"Epoch {epoch + 1}/{cfg['epochs']}")

        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for inputs, targets in train_loader_tqdm:
            inputs = inputs.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)  # (B,)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float()

                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == targets).sum().item()
                total_samples += targets.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_predictions / total_samples

        print(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg['save_model_name'])
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print("Early stopping triggered!")
                break

    return model

def evaluate_model(model, test_loader, cfg):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

total_start_time = time.time()
trained_model = fit_model(efficient_model, train_loader, val_loader, CFG)
total_end_time = time.time()

training_time = total_end_time - total_start_time
print(f"Total Training Time: {training_time // 3600:.0f}h {(training_time % 3600) // 60:.0f}m {training_time % 60:.0f}s")

test_loss, test_accuracy = evaluate_model(trained_model, test_loader, CFG)
print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")
