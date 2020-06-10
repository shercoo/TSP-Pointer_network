import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
import copy
import ptrnet3
import ptrnet4

class TSPdataset(Dataset):
    def __init__(self, filename, seq_len):
        super(TSPdataset, self).__init__()
        self.filename = filename
        self.seq_len = seq_len
        self.load_data()

    def load_data(self):
        f = open(self.filename, "r")
        data = []
        for line in f:
            input, tour_len = line.strip().split("output")
            input = list(map(float, input.strip().split(" ")))
            tour_len=float(tour_len)
            input = np.array(input).reshape((self.seq_len, 2))
            data.append((input, tour_len))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input, tour_len = self.data[index]
        return input, tour_len

SEQ_LEN=5
TEST_SIZE=256

dataset = TSPdataset("test_data/tsp_" + str(SEQ_LEN) + "testdata.txt", SEQ_LEN)
# dataset=TSPDataset(10000,SEQ_LEN)
dataloader = DataLoader(dataset, shuffle=True, batch_size=TEST_SIZE)

def beam_search(self,output,beam_size):
    batch_size=output.shape[0]
    seq_len=output.shape[1]
    lnpro=torch.log(output).data
    print(lnpro.size())
    ans=[]
    for case in range(batch_size):
        res=[([],0)]*beam_size
        for i in range(seq_len):
            # print("res",res)
            tmp=[]
            for nodes,prob in res:
                # print("nodes,prob",nodes,prob)
                for j in range(seq_len):
                    selected=False
                    if len(nodes)>0:
                        for node in nodes:
                            if node==j:
                                selected=True
                                break
                    if selected:
                        continue
                    next=copy.deepcopy(nodes)
                    next.append(j)
                    tmp.append((next,prob+lnpro[case][i][j]))
            res=sorted(tmp,key=lambda p: p[1],reverse=True)[0:beam_size]
        # print(res)
        ans.append(res[0][0])
    return ans


for input,TEST_SIZE in dataloader:
    net=ptrnet3.PtrNet(TEST_SIZE,2,ptrnet3.HIDDEN_DIM,ptrnet3.ENCODER_LAYERS);
    out=net(input)
    ans=beam_search().mean()
    print(ans)