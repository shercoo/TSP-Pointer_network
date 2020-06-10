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

        encoding_hidden_states = []
        W_1e = []
        hidden_state = torch.zeros([self.encoder_layers, self.batch_size, self.hidden_dim]).float()
        cell_state = torch.zeros([self.encoder_layers, self.batch_size, self.hidden_dim]).float()
        if USE_CUDA:
            hidden_state=hidden_state.cuda()
            cell_state=cell_state.cuda()

        # input-> seq_len*batch_size*input_dim
        input = input.transpose(0, 1).float()
        # print(input.size())
        # print(hidden_state.size())
        # print(cell_state.size())

        # encoding_hidden_states: seq_len * batch_size * hidden_dim
        # hidden_state & cell_state: encoder_layers * batch_size * hidden_dim
        encoding_hidden_states, (hidden_state, cell_state) = self.encoder(input, (hidden_state, cell_state))

        # W_1e: seq_len*batch_size*hidden_dim
        W_1e = self.ptr_W1(encoding_hidden_states)

        # encoding_hidden_states -> batch_size*seq_len*hidden_dim
        encoding_hidden_states = encoding_hidden_states.transpose(0, 1)
        # print(encoding_hidden_states.size())
        # encoding_hidden_states: batch_size*seq_len*hidden_dim

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
            # print("a_i", a_i[0])
            # print("hidden", encoding_hidden_states[0])

            # hidden_calced: batch_size*hidden_dim
            hidden_calced = torch.sum(torch.mul(a_i, encoding_hidden_states), 1)

            hidden_state = self.combine_hidden(torch.cat((hidden_calced, hidden_state), 1))

        # return: seq_len*batch_size*seq_len -> batch_size*seq_len*seq_len
        return torch.stack(output).transpose(0, 1)


class Trainer:
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

        self.optimizer = torch.optim.Adam(self.ptrNet.parameters(), lr=learning_rate)
        # for name,param in self.ptrNet.named_parameters():
        #     print(name,param.requires_grad)
        self.CEL = torch.nn.CrossEntropyLoss()
        self.episode = 0
        self.seq_len = 0
        self.filename = "5mydata.pt"
        if from_former_model:
            self.load_model()

    def train(self, input, ground_truth):
        self.seq_len = input.shape[1]
        batch_size=input.shape[0]
        output = self.ptrNet(input.float())
        # loss = torch.sqrt(torch.mean(torch.pow(output - truth, 2)))

        calc_output = output.reshape((batch_size * self.seq_len, self.seq_len))
        calc_ground_truth = ground_truth.reshape(-1)

        loss = self.CEL(calc_output, calc_ground_truth.long())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.episode += 1

        if self.episode % 5 == 0:
            print(output[0])
            print(ground_truth[0])
            print(loss.data)
            self.check_result(input, output, ground_truth)

        if self.episode % 200 == 0:
            self.save_model()

    def check_result(self, input, output, truth):
        # output:batch_size*seq_len*seq_len
        # truth:batch_size*seq_len
        tour_length = 0.0
        optimal_length = 0.0
        output = output.cpu().data.numpy()
        truth = truth.cpu().data.numpy()
        # print(input.shape)
        # print(output.shape)
        invalid_cnt = 0
        for case in range(input.shape[0]):
            tour = [np.argmax(output[case][i]) for i in range(input.shape[1])]
            flag = 0
            for i in range(input.shape[1]):
                for j in range(i + 1, input.shape[1]):
                    if tour[i] == tour[j]:
                        flag = 1
            invalid_cnt += flag
            if flag == 1:
                continue
            tmp_tour_length = 0.0
            tmp_optimal_length = 0.0
            for i in range(1, input.shape[1]):
                tmp_tour_length += torch.sqrt(torch.sum(torch.pow(input[case][tour[i]] - input[case][tour[i - 1]], 2)))
                tmp_optimal_length += torch.sqrt(
                    torch.sum(torch.pow(input[case][truth[case][i]] - input[case][truth[case][i - 1]], 2)))
            tmp_tour_length += torch.sqrt(
                torch.sum(torch.pow(input[case][tour[0]] - input[case][tour[input.shape[1] - 1]], 2)))
            tmp_optimal_length += torch.sqrt(
                torch.sum(torch.pow(input[case][truth[case][0]] - input[case][truth[case][input.shape[1] - 1]], 2)))
            if tmp_tour_length - tmp_optimal_length < -1e-6:
                print(input[case])
                print(tour)
                print(truth[case])
                print(tmp_tour_length, tmp_optimal_length)
            tour_length += tmp_tour_length
            optimal_length += tmp_optimal_length

        score = 0.0
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if np.argmax(output[i][j]) == truth[i][j]:
                    score += 1.0

        print(str(self.episode) + "th score: " + str(score / output.shape[0] / output.shape[1]))
        print(str(self.episode) + "th valid_ratio: " + str((input.shape[0] - invalid_cnt) / input.shape[0]))
        if input.shape[0] == invalid_cnt:
            print("No valid output!!!")
        else:
            print(str(self.episode) + "th length_ratio: " + str((tour_length / optimal_length).data))

    def save_model(self):
        torch.save(self.ptrNet.state_dict(), self.filename)
        print("Saved model")

    def load_model(self):
        self.ptrNet.load_state_dict(torch.load(self.filename))
        print("loaded model")


def get_one_hot_output(output):
    # output:batch_size*seq_len
    # one_hot:batch_size*seq_len*seq_len
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] -= 1
    return output
    # one_hot=np.zeros((output.shape[0],output.shape[1],output.shape[1]))
    # for i in range(output.shape[0]):
    #     for j in range(output.shape[1]):
    #         one_hot[i][j][output[i][j]-1]=1
    # return one_hot


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
            input = np.array(input).reshape((self.seq_len, 2))
            ground_truth = np.array(ground_truth)
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
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
ENCODER_LAYERS = 2
LOAD_FROM_EXISTED_MODEL = False

from Data_Generator import TSPDataset
dataset = TSPdataset("tsp_5-20_train/tsp_correct_" + str(SEQ_LEN) + ".txt", SEQ_LEN)
# dataset=TSPDataset(10000,SEQ_LEN)
dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

import warnings
warnings.filterwarnings("ignore",category=Warning)

trainer = Trainer(BATCH_SIZE,INPUT_DIM, HIDDEN_DIM, ENCODER_LAYERS, LEARNING_RATE, LOAD_FROM_EXISTED_MODEL)
from tqdm import tqdm

for i in range(MAX_EPOCHS):
    for input, ground_truth in dataloader:
        # if input.shape[0] != BATCH_SIZE:
        #     break
        # print(input.shape)
        # print(ground_truth.shape)
        # print(input)
        ground_truth = get_one_hot_output(ground_truth)
        if USE_CUDA:
            input = input.cuda()
            ground_truth = ground_truth.cuda()
        # print(ground_truth)
        trainer.train(input, ground_truth)

    # iterator = tqdm(dataloader, unit='Batch')
    # for i_batch, sample_batched in enumerate(iterator):
    #     iterator.set_description('Batch %i/%i' % (i+1, MAX_EPOCHS))
    #
    #     train_batch = torch.tensor(sample_batched['Points'])
    #     target_batch = torch.tensor(sample_batched['Solution'])
    #
    #     if USE_CUDA:
    #         train_batch = train_batch.cuda()
    #         target_batch = target_batch.cuda()
    #
    #     trainer.train(train_batch,target_batch)

    #
    # for i_batch, sample_batched in dataloader:
    #     print(type(i_batch))
    #     print(type(sample_batched))
    #     input = torch.tensor(sample_batched['Points'])
    #     ground_truth = torch.tensor(sample_batched['Solution'])
    #     # print(target_batch.shape)
    #     trainer.train(input,ground_truth)
