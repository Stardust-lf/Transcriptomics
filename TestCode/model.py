import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Utils import get_paths,get_inter_pos_frames
import numpy as np
import copy
import itertools

toTensor = lambda x: torch.tensor(x,dtype=torch.float32)
toArray = lambda x: np.array(x,dtype=np.float32)
# 定义超参数
input_size = 16616
output_size = 1
BATCH_SIZE = 16
NUM_EPOCHS = 100

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.LazyLinear(4096)
        self.fc2 = nn.LazyLinear(8192)
        self.fc3 = nn.LazyLinear(16616)
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
        self.fc1 = nn.LazyLinear(4096)
        self.fc2 = nn.LazyLinear(1024)
        self.fc3 = nn.LazyLinear(128)
        self.fc4 = nn.LazyLinear(8)
        self.fc5 = nn.LazyLinear(1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x



# 初始化网络和优化器
G = Generator(input_size, output_size)
D = Discriminator(output_size)
# criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
print(get_paths('Data/'))
slice1,slice2 = get_inter_pos_frames(get_paths('Data')[0],get_paths('Data')[1])
slice1_pos = slice1['Position']
slice2_pos = slice2['Position']
scale_max=[min(np.max(slice1['Position'][i][:,0]),np.max(slice2['Position'][i][:,0])) for i in range(len(slice1['Position']))]
scale_min=[max(np.min(slice1['Position'][i][:,0]),np.min(slice2['Position'][i][:,0])) for i in range(len(slice1['Position']))]
scale = list(zip(scale_min,scale_max))
slice1_pos = copy.deepcopy(slice1['Position'])
slice2_pos = copy.deepcopy(slice2['Position'])
values = [slice1_pos[0][i] for i in range(len(slice1_pos[0])) if scale[0][0]<= slice1_pos[0][i][0] <= scale[0][1]]
cutted_pos1 = []
cutted_pos2 = []
for n in range(len(slice1_pos)):
    cutted_pos1.append(np.array([slice1_pos[n][i] for i in range(len(slice1_pos[n])) if scale[n][0]<= slice1_pos[n][i][0] <= scale[n][1]]))
for n in range(len(slice2_pos)):
    cutted_pos2.append(np.array([slice2_pos[n][i] for i in range(len(slice2_pos[n])) if scale[n][0]<= slice2_pos[n][i][0] <= scale[n][1]]))

def wrap(position_set,scale=0.03):
    position_set += np.random.normal(0,scale,size=position_set.shape)

def wrapframe(slice_,scale=0.03):
    slice = copy.deepcopy(slice_)
    for i in range(len(slice)):
        wrap(slice[i],scale)
    return slice
def flatten_list_set(list_set):
    print(len(list_set))
    result = []
    for item in list_set:
        result+=item
    return np.array(result)

def generate_single_sample(slice):
    # warped = wrapframe(slice,wrap_scale*(NUM_EPOCHS-epoch)/NUM_EPOCHS)
    wrap_scale = np.random.randint(0,100)/100
    warped = wrapframe(slice, wrap_scale)
    input_ = toTensor(toArray(list(itertools.chain(*(warped + slice)))))
    input_ = toTensor(toArray(torch.flatten(input_)))
    return input_,1-wrap_scale
#训练网络
writer = SummaryWriter(log_dir='Run/')
for epoch in range(NUM_EPOCHS):
    print('-----------------------------------EPOCH{}-----------------------------------'.format(epoch))
    for i in range(100):
        # 训练判别器
        D.zero_grad()
        inputs = []
        scores = []
        for _ in range(BATCH_SIZE):
            wrap_f,score = generate_single_sample(cutted_pos1)
            inputs.append(wrap_f)
            scores.append((score))
        inputs = toTensor(toArray(inputs))
        scores = toTensor(toArray(scores))
        # real_data = wrapframe(cutted_pos1,0.03*(num_epochs-epoch)/num_epochs)
        # fake_data = wrapframe(cutted_pos1,0.3*(num_epochs-epoch)/num_epochs)
        # fix_data = cutted_pos1
        #
        # real_input = torch.tensor(np.array(list(itertools.chain(*(real_data + fix_data)))))
        # fake_input = torch.tensor(np.array(list(itertools.chain(*(fake_data + fix_data)))))
        # real_input = torch.flatten(real_input)
        # fake_input = torch.flatten(fake_input)
        # real_input = torch.tensor(np.array([real_input]),dtype=torch.float32)
        # fake_input = torch.tensor(np.array([fake_input]),dtype=torch.float32)
        outputs = D(inputs)

        loss = torch.sum(torch.abs(scores-outputs))
        writer.add_scalar('Loss D', loss, i)
        print('D Loss{}'.format(loss))
        loss.backward()
        D_optimizer.step()

        # 训练生成器
        G.zero_grad()
        G_input = toTensor(toArray(list(itertools.chain(*(slice2_pos + slice1_pos)))))
        G_input = toTensor(toArray(torch.flatten(G_input)))
        fake_outputs_GAN = G(G_input)

        G_loss_GAN = D(fake_outputs_GAN)
        writer.add_scalar('Loss G', G_loss_GAN, i)
        print('Generator Loss',G_loss_GAN)
        G_loss_GAN.backward()
        G_optimizer.step()
writer.close()

# 输出结果
#print(fake_data)
