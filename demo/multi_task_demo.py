# https://mp.weixin.qq.com/s/_wiQvTJvFBEccU_1Xm3d0w
# https://mp.weixin.qq.com/s/JqXogO4DQ8ybPm0uIAKS9A
import os
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset.components.my_dl import UTKFace
from models.HydraNet import HydraNet

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = UTKFace("../data/UTKFace")
    train_ds, val_ds = random_split(ds, [0.5, 0.5])
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=16, pin_memory=True, num_workers=0)
    val_dataloader = DataLoader(val_ds, shuffle=False, batch_size=16, pin_memory=True, num_workers=0)

    model = HydraNet().to(device=device)
    ethnicity_loss = nn.CrossEntropyLoss()
    gender_loss = nn.BCELoss()
    age_loss = nn.L1Loss()
    sig = nn.Sigmoid()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.09)
    n_epochs = 10

    for epoch in range(n_epochs):
        model.train()
        total_training_loss = 0
        for i, data in enumerate(tqdm(train_dataloader)):
            inputs = data["image"].to(device=device)
            age_label = data["age"].to(device=device)
            gender_label = data["gender"].to(device=device)
            eth_label = data["ethnicity"].to(device=device)
            optimizer.zero_grad()
            age_output, gender_output, eth_output = model(inputs)
            loss_1 = ethnicity_loss(eth_output, eth_label)
            loss_2 = gender_loss(sig(gender_output), gender_label.unsqueeze(1).float())
            loss_3 = age_loss(age_output, age_label.unsqueeze(1).float())
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()
            total_training_loss += loss

