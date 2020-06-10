import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PtrNet(nn.Module):
    def __init__(self, batch_size, seq_len, input_dim, hidden_dim):
        super(PtrNet, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.decoder = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        # same reference?
        self.ptr_W1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ptr_W2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ptr_v = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):
        # input: batch_size*seq_len*input_dim

        encoding_hidden_states = []
        W_1e = []
        hidden_state = torch.zeros([self.batch_size, self.hidden_dim])
        cell_state = torch.zeros([self.batch_size, self.hidden_dim])

        for i in range(self.seq_len):
            # current_input:batch_size*input_dim
            current_input=torch.tensor( input[:,i,:]).float()
            # print(current_input)

            hidden_state, cell_state = self.encoder(current_input, (hidden_state, cell_state))
            encoding_hidden_states.append(hidden_state)
            W_1e.append(self.ptr_W1(hidden_state))

        # W_1e: seq_len*batch_size*hidden_dim

        # encoding_hidden_states: seq_len*batch_size*hidden_dim -> batch_size*seq_len*hidden_dim
        encoding_hidden_states = torch.stack(encoding_hidden_states)
        encoding_hidden_states = torch.transpose(encoding_hidden_states, 0, 1)
        # print(encoding_hidden_states.size())
        # encoding_hidden_states: batch_size*seq_len*hidden_dim

        current_input = torch.full((self.batch_size, self.hidden_dim), -1.0)
        output = []
        for i in range(self.seq_len):
            u_i = []
            (hidden_state, cell_state) = self.decoder(current_input, (hidden_state, cell_state))
            for j in range(self.seq_len):
                #u_i.append( (batch_size*1)->batchsize )
                u_i.append(self.ptr_v(torch.tanh(W_1e[j] + self.ptr_W2(hidden_state))).squeeze(1))

            #u_i:seq_len*batch_size -> batch_size*seq_len
            u_i = torch.stack(u_i).t()
            # print("u_i",u_i.size())

            # a_i:batch_size*seq_len
            a_i = F.softmax(u_i, 1)
            # print("a_i",a_i.size())
            output.append(a_i)

            #a_i: batch_size*seq_len -> batch_size*seq_len*hidden_dim (copy)
            a_i = torch.unsqueeze(a_i, 2).expand(self.batch_size, self.seq_len, self.hidden_dim)
            # print("a_i",a_i.size())

            #hidden_state: batch_size*seq_len*hidden_dim -> batch_size*hidden_dim
            hidden_state = torch.sum(torch.mul(a_i, encoding_hidden_states), 1)
            # print("h",hidden_state.size())

        # return: seq_len*batch_size*seq_len -> batch_size*seq_len*seq_len
        return torch.stack(output).transpose(0, 1)


class Trainer:
    def __init__(self, batch_size, seq_len, input_dim, hidden_dim, learning_rate,from_former_model):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.ptrNet = PtrNet(batch_size, seq_len, input_dim, hidden_dim)
        self.optimizer = torch.optim.RMSprop(self.ptrNet.parameters(), self.learning_rate)
        self.episode = 0
        if from_former_model:
            self.load_model()

    def train(self, input, ground_truth):
        truth = torch.from_numpy(ground_truth)
        output = self.ptrNet(input)
        loss = torch.sqrt(torch.mean(torch.pow(output - truth, 2)))
        loss.backward()
        self.optimizer.step()
        self.episode += 1

        if self.episode % 10 == 0:
            print(output[0])
            print(ground_truth[0])
            self.check_correctness(output, truth)

        if self.episode % 200 == 0:
            self.save_model()

    def check_correctness(self, output, truth):
        # output:batch_size*seq_len*seq_len
        output = output.data.numpy()
        truth = truth.data.numpy()
        score = 0.0
        for i in range(output.shape[0]):
            for j in range(truth.shape[1]):
                if truth[i][j][np.argmax(output[i][j])] == 1:
                    score += 1.0

        print(str(self.episode) + "th score: " + str(score / output.shape[0] / output.shape[1]))

    def save_model(self):
        filename = "Models/tsp_net.pt"
        torch.save(self.ptrNet.state_dict(), filename)
        print("Saved model")

    def load_model(self):
        self.ptrNet.load_state_dict(torch.load('Models/tsp_net.pt'))
        print("loaded model")


def gen_sort_input(batch_size, seq_len):
    return np.random.rand(batch_size, seq_len, 1)


def gen_sort_output(input):
    # input:batch_size*seq_len*input_dim
    # output:batch_size*seq_len*seq_len
    output = np.zeros((input.shape[0], input.shape[1], input.shape[1]), dtype=np.float)
    for i in range(input.shape[0]):
        sorted_index = [idx for (idx, val) in sorted(enumerate(input[i]), key=lambda element: element[1][0])]
        for j in range(input.shape[1]):
            output[i][j][sorted_index[j]] = 1
    return output

def get_one_hot_output(output):
    #output:batch_size*seq_len
    # one_hot:batch_size*seq_len*seq_len
    one_hot=np.zeros((output.shape[0],output.shape[1],output.shape[1]))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            one_hot[i][j][output[i][j]-1]=1
    return one_hot


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
            input, ground_truth = line.strip().split("output")
            input = list(map(float, input.strip().split(" ")))
            ground_truth = list(map(int, ground_truth.strip().split(" ")))[0:-1]
            input=np.array(input).reshape((self.seq_len,2))
            ground_truth=np.array(ground_truth)
            data.append((input, ground_truth))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input, ground_truth = self.data[index]
        return input, ground_truth


MAX_EPOCHS = 10000
SEQ_LEN = 5
INPUT_DIM = 2
HIDDEN_DIM = 512
BATCH_SIZE = 256
LEARNING_RATE = 0.0001


dataset = TSPdataset("tsp_5-20_train/tsp_all_len5.txt", 5)
dataloader=DataLoader(dataset,num_workers=2,batch_size=BATCH_SIZE)


trainer = Trainer(BATCH_SIZE, SEQ_LEN, INPUT_DIM, HIDDEN_DIM, LEARNING_RATE,False)
for i in range(MAX_EPOCHS):
    for input,ground_truth in dataloader:
        if input.shape[0]!=BATCH_SIZE:
            break
        # print(input.shape)
        # print(ground_truth.shape)
        # print(input)
        ground_truth = get_one_hot_output(ground_truth)
        # print(ground_truth)
        trainer.train(input, ground_truth)


# net = PtrNet(1, 4, 1, 128)
# input=np.array([[[1],[2],[3],[4]]])
# print(input.shape)
# output=net(input)
# print("output",output.size())
