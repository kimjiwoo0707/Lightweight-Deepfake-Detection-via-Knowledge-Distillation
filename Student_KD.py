import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score


CFG = {
    'train_dir': '/home/work/jiu/Deepfake/train',
    'val_dir': '/home/work/jiu/Deepfake/valid',
    'test_dir': '/home/work/jiu/Deepfake/test',
    "batch_size": 32,
    "img_size": 224,
    "resize_size": 256,
    "learning_rate": 1e-4,
    "epochs": 20,
    "patience": 10,
    "dropout": 0.3,
    "fc_hidden_dim": 512,
    "temperature": 4.0,
    "alpha": 0.4,
    "kd_loss_type": "mse",
    "teacher_model_path": "efficientb7_teacher.pth",
    "save_student_name": "resnet8_student_best.pth",
    "num_workers": 2,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG["seed"])

device = torch.device(CFG["device"])


train_tf = transforms.Compose([
    transforms.Resize(CFG["resize_size"]),
    transforms.CenterCrop(CFG["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize(CFG["resize_size"]),
    transforms.CenterCrop(CFG["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(CFG["train_dir"], transform=train_tf)
val_ds   = datasets.ImageFolder(CFG["val_dir"],   transform=eval_tf)
test_ds  = datasets.ImageFolder(CFG["test_dir"],  transform=eval_tf)

pin_memory = (device.type == "cuda")

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=pin_memory)
val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=pin_memory)
test_loader  = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=pin_memory)


def build_teacher(cfg):
    teacher = models.efficientnet_b7(pretrained=True)

    in_features = teacher.classifier[1].in_features
    teacher.classifier = nn.Sequential(
        nn.Linear(in_features, cfg["fc_hidden_dim"]),
        nn.ReLU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["fc_hidden_dim"], 1),
    )

    state = torch.load(cfg["teacher_model_path"], map_location=device)
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[Teacher load_state_dict] missing keys:", missing)
        print("[Teacher load_state_dict] unexpected keys:", unexpected)

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet8(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for st in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, st))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def kd_loss(student_logits, teacher_logits, temperature: float, kd_type: str):
    if kd_type.lower() == "mse":
        return nn.MSELoss()(student_logits / temperature, teacher_logits / temperature)

    if kd_type.lower() == "kl":
        eps = 1e-6
        p_t = torch.sigmoid(teacher_logits / temperature).clamp(eps, 1 - eps)
        p_s = torch.sigmoid(student_logits / temperature).clamp(eps, 1 - eps)

        kl = p_t * torch.log(p_t / p_s) + (1 - p_t) * torch.log((1 - p_t) / (1 - p_s))
        return kl.mean()

    raise ValueError(f"Unknown kd_loss_type: {kd_type}")


def distillation_loss(student_logits, teacher_logits, targets, temperature, alpha, kd_type):
    hard = nn.BCEWithLogitsLoss()(student_logits.squeeze(1), targets)
    soft = kd_loss(student_logits, teacher_logits, temperature, kd_type)
    return alpha * soft + (1 - alpha) * hard


def train_kd(student, teacher, train_loader, val_loader, cfg):
    optimizer = optim.Adam(student.parameters(), lr=cfg["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, verbose=True
    )

    best_val = float("inf")
    patience_cnt = 0

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        student.train()

        train_loss_sum = 0.0
        for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()

            optimizer.zero_grad()

            with torch.no_grad():
                t_logits = teacher(inputs)

            s_logits = student(inputs)
            loss = distillation_loss(
                s_logits, t_logits, targets,
                cfg["temperature"], cfg["alpha"], cfg["kd_loss_type"]
            )

            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)

        student.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).float()

                t_logits = teacher(inputs)
                s_logits = student(inputs)

                vloss = distillation_loss(
                    s_logits, t_logits, targets,
                    cfg["temperature"], cfg["alpha"], cfg["kd_loss_type"]
                )
                val_loss_sum += vloss.item()

        val_loss = val_loss_sum / len(val_loader)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(student.state_dict(), cfg["save_student_name"])
            print(f"Saved best student checkpoint -> {cfg['save_student_name']}")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print("Early stopping triggered!")
                break

@torch.no_grad()
def evaluate(student, dataloader):
    student.eval()

    all_targets = []
    all_preds = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for inputs, targets in tqdm(dataloader, desc="Testing", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        logits = student(inputs).squeeze(1)         # (B,)
        probs = torch.sigmoid(logits)               # (B,)
        preds = (probs > 0.5).float()

        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)

    acc = (all_targets_np == all_preds_np).mean()

    precision = precision_score(all_targets_np, all_preds_np, zero_division=0)
    recall = recall_score(all_targets_np, all_preds_np, zero_division=0)
    f1 = f1_score(all_targets_np, all_preds_np, zero_division=0)

    total_time = end - start
    fps = len(dataloader.dataset) / total_time if total_time > 0 else float("inf")

    return acc, precision, recall, f1, fps


teacher = build_teacher(CFG)
student = ResNet8(num_classes=1).to(device)

print("Training Student with Knowledge Distillation...")
train_kd(student, teacher, train_loader, val_loader, CFG)

print("\nLoading best student checkpoint for test...")
student.load_state_dict(torch.load(CFG["save_student_name"], map_location=device))
student.to(device)

acc, precision, recall, f1, fps = evaluate(student, test_loader)
print("\nFinal Test Metrics")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"FPS      : {fps:.2f}")
