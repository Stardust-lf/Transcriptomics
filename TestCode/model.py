import torch
import torch.nn as nn
from Utils import get_paths,get_inter_pos_frames
import numpy as np
# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 定义超参数
input_size = 100
output_size = 784
batch_size = 64
num_epochs = 100

# 初始化网络和优化器
G = Generator(input_size, output_size)
D = Discriminator(output_size)
criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
print(get_paths('Data/'))
slice1,slice2 = get_inter_pos_frames(get_paths('Data')[0],get_paths('Data')[1])
slice1_pos = slice1['Position']
def wrap(position_set,scale=0.03):
    position_set += np.random.normal(0,scale,size=position_set.shape)

def wrapframe(slice,scale=0.03):
    for i in range(len(slice)):
        wrap(slice1[i],scale)
#训练网络
for epoch in range(num_epochs):
    for i in range(30):
        # 训练判别器
        D.zero_grad()
        real_data = wrapframe(slice1_pos,0.03) + slice1_pos
        fake_data = wrapframe(slice1_pos,0.3) + slice1_pos

        real_outputs = D(real_data)
        real_loss = -np.log(real_outputs)

        fake_outputs = D(fake_data)
        fake_loss = -np.log(1-fake_outputs)

        D_loss = real_loss + fake_loss
        D_loss.backward()
        D_optimizer.step()

        # 训练生成器
        # G.zero_grad()
        # fake_outputs_GAN = D(fake_data)
        # G_loss_GAN = criterion(fake_outputs_GAN, real_labels)
        #
        # G_loss_GAN.backward()
        # G_optimizer.step()


# 输出结果
print(fake_data)
