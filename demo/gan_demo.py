import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


# 定义训练过程
def train_gan(generator, discriminator, dataloader, num_epochs=50, lr=0.0002):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for real_data, _ in dataloader:
            batch_size = real_data.size(0)
            real_data = real_data.view(-1, 28 * 28)

            # 训练判别器
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            real_output = discriminator(real_data)
            real_loss = criterion(real_output, real_labels)
            real_loss.backward()

            noise = torch.randn(batch_size, 100)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data.detach())
            fake_loss = criterion(fake_output, fake_labels)
            fake_loss.backward()

            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            output = discriminator(fake_data)
            gen_loss = criterion(output, real_labels)
            gen_loss.backward()
            optimizer_G.step()

        # 打印损失
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Generator Loss: {gen_loss.item():.4f}, "
              f"Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}")

if __name__ == "__main__":
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    # 创建生成器和判别器
    generator = Generator(input_size=100, output_size=28*28)
    discriminator = Discriminator(input_size=28*28)

    # 训练GAN
    train_gan(generator, discriminator, dataloader)

    # 生成新样本并显示
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, 100)
        generated_samples = generator(noise).view(-1, 28, 28).numpy()

    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_samples[i], cmap='gray')
        plt.axis('off')
    plt.show()
