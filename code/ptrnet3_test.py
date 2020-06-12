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
import argparse

parser=argparse.ArgumentParser(description="Basic Pointer Network Tester.")
parser.add_argument('--seq_len', default=10, type=int, choices=[5,10,20,40,50])
args=vars(parser.parse_args())

SEQ_LEN = args['seq_len']
MAX_EPOCHS = 1
INPUT_DIM = 2
HIDDEN_DIM = 512
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
ENCODER_LAYERS = 2
LOAD_FROM_EXISTED_MODEL = True


if torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

class PtrNet(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, encoder_layers):
        super(PtrNet, self).__init__()

        self.batch_size = 0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.seq_len = 0

        self.encoder = nn.LSTM(self.input_dim, self.hidden_dim, self.encoder_layers)
        self.decoder = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.ptr_W1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ptr_W2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ptr_v = nn.Linear(self.hidden_dim, 1)
        self.combine_hidden = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, input):

        # input: batch_size*seq_len*input_dim
        input = torch.tensor(input)
        self.batch_size=input.shape[0]
        self.seq_len = input.shape[1]

        hidden_state = torch.zeros([self.encoder_layers, self.batch_size, self.hidden_dim]).float()
        cell_state = torch.zeros([self.encoder_layers, self.batch_size, self.hidden_dim]).float()
        if USE_CUDA:
            hidden_state=hidden_state.cuda()
            cell_state=cell_state.cuda()

        # input-> seq_len*batch_size*input_dim
        input = input.transpose(0, 1).float()

        # encoding_hidden_states: seq_len * batch_size * hidden_dim
        # hidden_state & cell_state: encoder_layers * batch_size * hidden_dim
        encoding_hidden_states, (hidden_state, cell_state) = self.encoder(input, (hidden_state, cell_state))

        # W_1e: seq_len*batch_size*hidden_dim
        W_1e = self.ptr_W1(encoding_hidden_states)

        # encoding_hidden_states -> batch_size*seq_len*hidden_dim
        encoding_hidden_states = encoding_hidden_states.transpose(0, 1)

        current_input = torch.full((self.batch_size, self.input_dim), -1.0)
        if USE_CUDA:
            current_input=current_input.cuda()

        # hidden_state & cell_state-> batch_size * hidden_dim
        hidden_state = hidden_state[-1]
        cell_state = cell_state[-1]

        # input-> batch_size*seq_len*input_dim
        input = input.transpose(0, 1)
        output = []

        for i in range(self.seq_len):
            u_i = []
            (hidden_state, cell_state) = self.decoder(current_input, (hidden_state, cell_state))
            for j in range(self.seq_len):
                # u_i.append( (batch_size*1)->batchsize )
                u_i.append(self.ptr_v(torch.tanh(W_1e[j] + self.ptr_W2(hidden_state))).squeeze(1))

            # u_i-> batch_size*seq_len
            u_i = torch.stack(u_i).t()

            # a_i:batch_size*seq_len
            a_i = F.softmax(u_i, 1)
            output.append(a_i)

            # chosen_value:batch_size
            chosen_value = a_i.argmax(1)

            # current_input: batch_size*input_dim
            current_input = [input[i][chosen_value[i]] for i in range(self.batch_size)]
            current_input = torch.stack(current_input)

            # a_i: batch_size*seq_len -> batch_size*seq_len*hidden_dim (same data)
            a_i = a_i.unsqueeze(2).expand(self.batch_size, self.seq_len, self.hidden_dim)

            # hidden_calced: batch_size*hidden_dim
            hidden_calced = torch.sum(torch.mul(a_i, encoding_hidden_states), 1)

            hidden_state = self.combine_hidden(torch.cat((hidden_calced, hidden_state), 1))

        # return: seq_len*batch_size*seq_len -> batch_size*seq_len*seq_len
        return torch.stack(output).transpose(0, 1)

def beam_search(output,beam_size):
    batch_size=output.shape[0]
    seq_len=output.shape[1]
    lnpro=torch.log(output).data
    # print(lnpro.size())
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



class Tester:
    def __init__(self, batch_size,input_dim, hidden_dim, encoder_layers, learning_rate, from_former_model):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.ptrNet = PtrNet(batch_size, input_dim, hidden_dim, encoder_layers)

        # self.ptrNet=PointerNet(128,hidden_dim,encoder_layers,0.,False)
        if USE_CUDA:
            self.ptrNet.cuda()
            net = torch.nn.DataParallel(self.ptrNet, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        self.episode=0
        self.tot_ans=0.0
        self.tot_len=0.0
        self.seq_len = 0
        self.filename = "../model/" + str(20 if SEQ_LEN>=20 else SEQ_LEN) + "mydata.pt"
        if from_former_model:
            self.load_model()

    def test(self, input, optimal_len):
        self.seq_len = input.shape[1]
        output = self.ptrNet(input.float())
        ans=self.calc_len(input,output)
        optimal_len=optimal_len.mean()
        self.episode+=1
        self.tot_ans+=ans
        self.tot_len+=optimal_len
        print(self.episode,ans.data.numpy(),optimal_len.data.numpy())
        print(self.tot_ans/self.tot_len)

    def calc_len(self, input, output):
        # output:batch_size*seq_len*seq_len
        # truth:batch_size*seq_len
        batch_size=input.shape[0]
        seq_len=input.shape[1]

        ans_length = 0.0

        ans = np.array( beam_search(output.cpu(), 2))

        for case in range(batch_size):
            for i in range(1, seq_len):
                ans_length += torch.sqrt(torch.sum(torch.pow(input[case][ans[case][i]] - input[case][ans[case][i - 1]], 2)))
            ans_length += torch.sqrt(
                torch.sum(torch.pow(input[case][ans[case][0]] - input[case][ans[case][seq_len - 1]], 2)))

        return ans_length/batch_size;


    def load_model(self):
        self.ptrNet.load_state_dict(torch.load(self.filename,map_location=torch.device('cpu')))
        print("loaded model")


def get_one_hot_output(output):
    # output:batch_size*seq_len
    # one_hot:batch_size*seq_len*seq_len
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] -= 1
    return output


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



dataset = TSPdataset("../test_data/tsp" + str(SEQ_LEN) + "_testdata.txt", SEQ_LEN)
dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

import warnings
warnings.filterwarnings("ignore",category=Warning)

tester = Tester(BATCH_SIZE,INPUT_DIM, HIDDEN_DIM, ENCODER_LAYERS, LEARNING_RATE, LOAD_FROM_EXISTED_MODEL)

for i in range(MAX_EPOCHS):
    for input, optimal_len in dataloader:
        if USE_CUDA:
            input = input.cuda()
        # print(ground_truth)
        tester.test(input,optimal_len)

print(tester.tot_ans/tester.episode,tester.tot_len/tester.episode)

