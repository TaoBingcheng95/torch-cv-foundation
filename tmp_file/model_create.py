from torchvision import models
from torch import nn
import torch
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, xb):
        return self.network(xb)

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))



class MyModel(ImageClassificationBase):
    def __init__(self):
        super().__init__(ImageClassificationBase)
        self.model = model_vgg19

    def forward(self, xb):
        return self.model(xb)


if __name__ == '__main__':

    model_vgg19 = models.vgg19(weights=None)
    model_resnet50 = models.resnet50(weights=None)

    for name, param in model_vgg19.named_parameters():
        # print(name, param.requires_grad)
        param.requires_grad = False

    for name, param in model_resnet50.named_parameters():
        # print(name, param.requires_grad)
        param.requires_grad = False

    print(model_vgg19.classifier)
    # 我们可以看到这个预训练模型是为对1000个类进行分类而设计的，但是我们只需要 10 类分类，所以稍微改变一下这个模型。
    our_classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=2048, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5, inplace=False),
                                   nn.Linear(in_features=2048, out_features=1024, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5, inplace=False),
                                   nn.Linear(in_features=1024, out_features=10, bias=True),
                                   nn.LogSoftmax(dim=1))