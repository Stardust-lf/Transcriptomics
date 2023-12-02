import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Utils import get_paths,get_inter_pos_frames,wrapping_simu
import numpy as np
import copy
import itertools
import seaborn as sns
import pickle
toTensor = lambda x: torch.tensor(x,dtype=torch.float32)
toArray = lambda x: np.array(x,dtype=np.float32)
toFlatten = lambda x: toArray(list(itertools.chain(*x)))

input_size = 16616
output_size = 1
BATCH_SIZE = 16
NUM_EPOCHS = 10

class Generator(nn.Module):
    def __init__(self, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.LazyLinear(2048)
        self.fc2 = nn.LazyLinear(4096)
        self.fc3 = nn.LazyLinear(output_size)
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


slice1,slice2 = get_inter_pos_frames(get_paths('Data')[0],get_paths('Data')[1])
slice1_pos = slice1['Position']
slice2_pos = slice2['Position']
scale_max=[min(np.max(slice1['Position'][i][:,0]),np.max(slice2['Position'][i][:,0])) for i in range(len(slice1['Position']))]
scale_min=[max(np.min(slice1['Position'][i][:,0]),np.min(slice2['Position'][i][:,0])) for i in range(len(slice1['Position']))]
scale = list(zip(scale_min,scale_max))
slice1_pos = copy.deepcopy(slice1['Position'])
slice2_pos = copy.deepcopy(slice2['Position'])
values = [slice1_pos[0][i] for i in range(len(slice1_pos[0])) if scale[0][0]<= slice1_pos[0][i][0] <= scale[0][1]]
filled_pos1 = []
filled_pos2 = []
for n in range(len(slice1_pos)):
    leng1 = len(slice1_pos[n])
    leng2 = len(slice2_pos[n])
    if leng1>leng2:
        slice2_pos[n] = np.concatenate([slice2_pos[n],np.zeros(shape=[leng1-leng2,2])],axis=0)
    elif leng1<leng2:
        slice1_pos[n] = np.concatenate([slice1_pos[n],np.zeros(shape=[leng2-leng1,2])],axis=0)
# for n in range(len(slice1_pos)):
#     cutted_pos1.append(np.array([slice1_pos[n][i] for i in range(len(slice1_pos[n])) if scale[n][0]<= slice1_pos[n][i][0] <= scale[n][1]]))
# for n in range(len(slice2_pos)):
#     cutted_pos2.append(np.array([slice2_pos[n][i] for i in range(len(slice2_pos[n])) if scale[n][0]<= slice2_pos[n][i][0] <= scale[n][1]]))
cutted_pos1 = toTensor(toFlatten(slice1_pos))
cutted_pos2 = toTensor(toFlatten(slice2_pos))
# def wrap(position_set,scale=0.03):
#     position_set += np.random.normal(0,scale,size=position_set.shape)

def wrapframe(slice_):
    slice_,score = wrapping_simu(slice_)
    return slice_,score
def flatten_list_set(list_set):
    result = []
    for item in list_set:
        result+=item
    return np.array(result)

def generate_single_sample(slice):
    # warped = wrapframe(slice,wrap_scale*(NUM_EPOCHS-epoch)/NUM_EPOCHS)
    wrapped = wrapframe(slice)
    #input_ = toTensor(toArray(list(itertools.chain(*(warped + slice)))))
    #input_ = toTensor(toArray(torch.flatten(input_)))
    return wrapped

# 初始化网络和优化器
G = Generator(np.size(toFlatten(slice1_pos)))
D = Discriminator(np.size(toFlatten(slice1_pos)))
# criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.03)

#训练网络
#writer = SummaryWriter(log_dir='Run/')
#history = []
lossD = []
lossG = []
for epoch in range(NUM_EPOCHS):
    print('-----------------------------------EPOCH{}-----------------------------------'.format(epoch))
    for i in range(5):
        D.zero_grad()
        samples_and_scores = [generate_single_sample(cutted_pos1) for i in range(BATCH_SIZE)]

        samples, scores = zip(*samples_and_scores)
        scores = toTensor(scores)
        slice_wrapped = toTensor([toFlatten(sample) for sample in samples]).reshape(BATCH_SIZE, -1)
        slice_true = toTensor(cutted_pos1).reshape(-1).repeat(BATCH_SIZE,1)
        flows = slice_true-slice_wrapped

        D_input = torch.concatenate([slice_true, flows], dim=1)
        outputs = D(D_input)

        loss = torch.mean(torch.abs(scores-outputs))
        lossD_num = loss.detach().numpy()
        print('D Loss',lossD_num)
        lossD.append(lossD_num)
        #writer.add_scalar('Loss D', loss, i)
        #history.append(loss.detach().numpy())
        #print('D Loss{}'.format(loss))
        loss.backward()
        D_optimizer.step()


    for _ in range(1):
        G.zero_grad()
        fake_flow_GAN = G(toTensor(cutted_pos2).reshape(1,-1))
        G_loss_GAN = D(torch.concatenate([toTensor(cutted_pos1).reshape(1,-1), fake_flow_GAN], dim=1))
        #writer.add_scalar('Loss G', G_loss_GAN, i)
        #fig,ax = plt.subplots(figsize=(6,6),dpi=150)
        # fake_slice = (cutted_pos2 + fake_flow_GAN.reshape(-1,2)).detach().numpy()
        # plt.scatter(x=fake_slice[:,0],y=fake_slice[:,1],s=3)
        # plt.show()
        #writer.add_figure('my_image',fig,i)
        print('G Loss',G_loss_GAN)
        lossG.append(G_loss_GAN.detach().numpy())
        G_loss_GAN.backward()
        G_optimizer.step()
plt.plot(lossD)
plt.plot(lossG)
plt.show()
# f = open('G_loss','wb')
# pickle.dump(toArray(lossG), f)
# f.close()
#writer.close()

# 输出结果
#print(fake_data)
