import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from IPython.display import clear_output
import matplotlib.pyplot as plt
def getset(citynumber,samples):
    torch.manual_seed(66)
    data_set = []
    for l in range(samples):
        #生成在坐标在0 1 之间的
        x = torch.FloatTensor(2, citynumber).uniform_(0, 1)
        data_set.append(x)
    return data_set

#就是算一下路径总的长度
def reward(result):
    batch_size=result[0].size(0)
    length=Variable(torch.zeros([batch_size]))
    for i in range(len(result)-1):
        length+=torch.norm(result[i]-result[i+1],dim=1)
    length+=torch.norm(result[0]-result[len(result)-1],dim=1)
    return length
class attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False):
        super(attention, self).__init__()
        
        self.use_tanh = use_tanh
        self.C = 10
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_ref   = nn.Conv1d(hidden_size, hidden_size, 1, 1)
        V = torch.FloatTensor(hidden_size)
        self.V = nn.Parameter(V)
        self.V.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))
    
    def forward(self,query,ref):
        #参考了bahdanau attention机制,也就是pointernetwork中的attention机制
        #query batchsize*hidden_size
        #ref   batchsize*seq_len*hiddensize
        batch_size = ref.size(0)
        seq_len    = ref.size(1)
        ref = ref.permute(0, 2, 1)
        query = self.W_query(query).unsqueeze(2)  # batch_size x hidden_size x 1
        ref   = self.W_ref(ref)  # batch_size x hidden_size x seq_len
        equery = query.repeat(1, 1, seq_len) #batch_size x hidden_size x seq_len
        V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) #batch_size x 1 x hidden_size
        logits = torch.bmm(V, F.tanh(equery + ref)).squeeze(1)
        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits  
        return ref, logits
class Embedding(nn.Module):
    #
    def __init__(self,in_size,embedding_size):
        super(Embedding,self).__init__()
        self.embedding_size=embedding_size
        self.embedding=nn.Parameter(torch.FloatTensor(in_size,embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
    def forward(self,inputs):
        batch_size=inputs.size(0)
        seq_len=inputs.size(2)
        embedding=self.embedding.repeat(batch_size,1,1)
        res=[]
        inputs=inputs.unsqueeze(1)
        for i in range(seq_len):
            res.append(torch.bmm(inputs[:,:,:,i].float(),embedding))
        return torch.cat(res,1)

class ptrNet(nn.Module):
    def __init__(self,embedding_size,hidden_size,seq_len,n_glimpses,tanh_num,use_tanh):
        super(ptrNet,self).__init__()

        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.n_glimpses=n_glimpses
        self.seq_len=seq_len
        self.embedding=Embedding(2,embedding_size)
        self.encoder=nn.LSTM(embedding_size,hidden_size,batch_first=True)
        self.decoder=nn.LSTM(embedding_size,hidden_size,batch_first=True)
        self.pointer=attention(hidden_size,use_tanh=use_tanh)
        self.glimpse=attention(hidden_size,use_tanh=False)
        self.decoder_input=nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
    def forward(self,inputs):
        #inputs  batch_size * 1 * seqlen
        batch_size=inputs.size(0)
        seq_len=inputs.size(2)
        embedded_input=self.embedding(inputs)
        #print(embedded_input.size())
        encoder_output,(hidden,context)=self.encoder(embedded_input)
        #print(encoder_output.size())
        probs=[]
        prev_index=[]
        mask = torch.zeros(batch_size, seq_len).byte()
        index=None
        decoder_inputs=self.decoder_input.unsqueeze(0).repeat(batch_size,1)
        for i in range(seq_len):
            temp,(hidden,context)=self.decoder(decoder_inputs.unsqueeze(1),(hidden,context))
            query=hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref,logits=self.glimpse(query,encoder_output)
                mc=mask.clone()
                if index is not None:
                    mc[[i for i in range(logits.size(0))],index.data]=1
                    logits[mc]=-np.inf
                mask=mc
                query=torch.bmm(ref,F.softmax(logits).unsqueeze(2)).squeeze(2)
            temp,logits=self.pointer(query,encoder_output)
            #print(logits.size())
            
            mc=mask.clone()
            if index is not None:
                mc[[i for i in range(logits.size(0))],index.data]=1
                logits[mc]=-np.inf
            mask=mc
            #policy gradient 通过reward对选择行为的可能性影响，增加好的行为选中概率
            prob=F.softmax(logits)
            #print(prob.size())
            index=prob.multinomial(1).squeeze(1)
            decoder_inputs=embedded_input[[i for i in range(batch_size)],index.data,:]
            probs.append(prob)
            prev_index.append(index)
        return probs,prev_index

class rlmodel(nn.Module):
    def __init__(self,embedding_size,hidden_size,seq_len,n_glimpses,tanh_num,use_tanh,reward):
        super(rlmodel,self).__init__()
        self.reward=reward
        self.net=ptrNet(
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_num,
            use_tanh
        )
    def forward(self,inputs):
        #inputs batch_size  input_size  seq_len
        batch_size=inputs.size(0)
        input_size=inputs.size(1)
        seq_len=inputs.size(2)
        probs,index=self.net(inputs)
        route=[]
        inputs=inputs.transpose(1,2)
        #print(inputs.size())
        for i in index:
            route.append(inputs[[x for x in range(batch_size)],i.data,:])
        
        #print(route.size())
        route_prob=[]
        for prob,ind in zip(probs,index):
            route_prob.append(prob[[x for x in range(batch_size)],ind.data])
        
        route_reward=self.reward(route)
        return route_reward,route_prob,route,index

class trainer:
    def __init__(self,model,train_data):
        self.model=model
        self.train_data=train_data
        self.batch_size=BATCH_SIZE
        self.train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
        self.optim=optim.Adam(model.net.parameters(),lr=1e-4)
        self.max_grad_norm=MAX_GRAD_NORM
        self.train_routeset=[]
        self.epochs=0
        #self.epoch
    def train(self,epoch):
        critic_ave=torch.zeros(1)
        for epo in range(epoch):
            for i,batch in enumerate(self.train_loader):
                logprob=0
                self.model.train()
                #active search
                #sample input
                inputs=Variable(batch)
                #print(inputs.size())
                route_reward,probs,route,route_index=self.model(inputs)
                if i==0:
                    critic_ave=route_reward.mean()
                else:
                    critic_ave=critic_ave*BETA+(1.0-BETA)*route_reward.mean()
                improve=route_reward-critic_ave
                critic_ave=critic_ave.detach()
                for prob in probs:
                    logprob+=torch.log(prob)
                logprob[logprob<-1000]=0
                #monte carlo sampling function
                Reinfor=improve*logprob
                loss=Reinfor.mean()
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.net.parameters(),float(self.max_grad_norm),norm_type=2)
                self.optim.step()
                self.train_routeset.append(route_reward.mean().item())
                if i % 10 == 0:
                    self.plot(self.epochs)
            self.epochs += 1
    def plot(self, epoch):
        clear_output(True)
        plt.figure(figsize=(30,5))
        plt.subplot(121)
        plt.plot(self.train_routeset)
        plt.grid()
        plt.show()
        



train_size=100000
trainset=getset(20,train_size)
EMBEDDING_SIZE=128
BATCH_SIZE=128
HIDDEN_SIZE=128
TANH_NUM=20
USE_TANH=True
BETA=0.9
MAX_GRAD_NORM=2

my_model=rlmodel(EMBEDDING_SIZE,HIDDEN_SIZE,20,1,TANH_NUM,USE_TANH,reward)
tsp=trainer(my_model,trainset)
tsp.train(1000)


                