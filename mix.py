import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class SBTEncoder(nn.Module):
    def __init__(self,config):
        super(SBTEncoder,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.embedding_t = nn.Embedding(config.DICT_SIZE_2,config.MODEL_SIZE)
        self.embedding_v = nn.Embedding(config.DICT_SIZE_1,config.MODEL_SIZE)
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_{}".format(i),
            nn.LSTM(config.MODEL_SIZE,config.MODEL_SIZE))
    
    def forward(self,inputs):
        device = self.device
        config = self.config
        in1, in2 = inputs
        lengths = [len(x) for x in in1]
        in1 = [torch.tensor(x).to(device) for x in in1]
        in1 = rnn_utils.pad_sequence(in1)
        tensor1 = self.embedding_t(in1)
        in2 = [torch.tensor(x).to(device) for x in in2]
        in2 = rnn_utils.pad_sequence(in2)
        tensor2 = self.embedding_v(in2)
        tensor = tensor1 + tensor2
        for i in range(config.NUM_LAYER):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)
            tensor, (h, c) = getattr(self,"layer_{}".format(i))(tensor)
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip

        cx = c
        hx = h
        ys = [y[:i] for y,i in zip(torch.unbind(tensor,axis=1),lengths)]
        return ys, (hx, cx)

class NormalEncoder(nn.Module):
    def __init__(self,config):
        super(NormalEncoder,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.embedding = nn.Embedding(config.DICT_SIZE_1,config.MODEL_SIZE)
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_{}".format(i),
            nn.LSTM(config.MODEL_SIZE,config.MODEL_SIZE))
    
    def forward(self,inputs):
        device = self.device
        config = self.config
        lengths = [len(x) for x in inputs]
        inputs = [torch.tensor(x).to(device) for x in inputs]
        inputs = rnn_utils.pad_sequence(inputs)
        tensor = self.embedding(inputs)
        for i in range(config.NUM_LAYER):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)
            tensor, (h, c) = getattr(self,"layer_{}".format(i))(tensor)
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip

        cx = c
        hx = h
        ys = [y[:i] for y,i in zip(torch.unbind(tensor,axis=1),lengths)]
        return ys, (hx, cx)

class Attn(nn.Module):
    def __init__(self,config):
        super(Attn,self).__init__()
        self.config = config
        self.Q = nn.Linear(config.MODEL_SIZE,config.MODEL_SIZE)
        self.K = nn.Linear(config.MODEL_SIZE,config.MODEL_SIZE)
        self.V = nn.Linear(config.MODEL_SIZE,config.MODEL_SIZE)
        self.W = nn.Linear(config.MODEL_SIZE,1)

    def forward(self,q,k,v,mask):
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        q = q.unsqueeze(1)
        k = k.unsqueeze(0)
        attn_weight = self.W(torch.tanh(q+k))
        attn_weight = attn_weight.squeeze(-1)
        attn_weight = torch.where(mask,attn_weight,torch.tensor(-1e6).to(q.device))
        attn_weight = attn_weight.softmax(1)
        attn_weight = attn_weight.unsqueeze(-1)
        context = attn_weight*v
        context = context.sum(1)
        return context

class Decoder(nn.Module):    
    def __init__(self,config):
        super(Decoder,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.attn1 = Attn(config)
        self.attn2 = Attn(config)
        self.dropout1 = nn.Dropout(config.DROPOUT)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.dropout3 = nn.Dropout(config.DROPOUT)
        self.dropout4 = nn.Dropout(config.DROPOUT)
        self.embedding = nn.Embedding(config.DICT_SIZE_1,config.MODEL_SIZE)
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_{}".format(i),
            nn.LSTM(config.MODEL_SIZE,config.MODEL_SIZE))
        self.lstm = nn.LSTM(2*config.MODEL_SIZE,config.MODEL_SIZE)
        self.fc = nn.Linear(config.MODEL_SIZE,config.DICT_SIZE_1)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self,inputs,l_states,enc1,mask1,enc2,mask2):
        config = self.config
        tensor = self.embedding(inputs)
        for i in range(config.NUM_LAYER):
            skip = tensor
            tensor, l_states[i] = getattr(self,"layer_{}".format(i))(tensor,l_states[i])
            tensor = tensor + skip
        
        context1 = self.attn1(tensor,enc1,enc1,mask1)
        context2 = self.attn2(tensor,enc2,enc2,mask2)
        context = context1 + context2
        tensor = torch.cat([tensor,context],-1)
        tensor, l_states[-1] = self.lstm(tensor,l_states[-1])
        tensor = self.fc(tensor)

        return tensor, l_states

    def get_loss(self,enc1,enc2,states,targets):
        device = self.device
        config = self.config

        targets = [torch.tensor([config.START_TOKEN]+x+[config.END_TOKEN]).to(device) for x in targets]
        targets = rnn_utils.pad_sequence(targets)
        inputs = targets[:-1]
        targets = targets[1:]

        mask1 = [torch.ones(x.shape[0]).to(device) for x in enc1]
        mask1 = rnn_utils.pad_sequence(mask1)
        mask1 = mask1.unsqueeze(0)
        mask1 = mask1.eq(1)
        enc1 = rnn_utils.pad_sequence(enc1)

        mask2 = [torch.ones(x.shape[0]).to(device) for x in enc2]
        mask2 = rnn_utils.pad_sequence(mask2)
        mask2 = mask2.unsqueeze(0)
        mask2 = mask2.eq(1)
        enc2 = rnn_utils.pad_sequence(enc2)
        
        h,c = states
        enc1 = self.dropout1(enc1)
        enc2 = self.dropout2(enc2)
        h = self.dropout3(h)
        c = self.dropout4(c)
        l_states = [(h,c) for _ in range(config.NUM_LAYER+1)]

        tensor, l_states = self.forward(inputs,l_states,enc1,mask1,enc2,mask2)

        tensor = tensor.reshape(-1,config.DICT_SIZE_1)
        targets = targets.reshape(-1)
        loss = self.loss_function(tensor,targets)

        return loss

    def translate(self,enc1,enc2,states):
        device = self.device
        config = self.config
        h,c = states

        lengths1 = [x.shape[0] for x in enc1]
        mask1 = [torch.ones(x).to(device) for x in lengths1]
        mask1 = rnn_utils.pad_sequence(mask1)
        mask1 = mask1.unsqueeze(0)
        mask1 = mask1.eq(1)
        enc1 = rnn_utils.pad_sequence(enc1)

        l_states = [(h,c) for _ in range(config.NUM_LAYER+1)]
        
        lengths2 = [x.shape[0] for x in enc2]
        mask2 = [torch.ones(x).to(device) for x in lengths2]
        mask2 = rnn_utils.pad_sequence(mask2)
        mask2 = mask2.unsqueeze(0)
        mask2 = mask2.eq(1)
        enc2 = rnn_utils.pad_sequence(enc2)
        
        preds = [[config.START_TOKEN] for _ in range(len(lengths1))]
        dec_input = torch.tensor(preds).to(device).view(1,-1)
        for _ in range(config.MAX_OUTPUT_SIZE):
            tensor, l_states = self.forward(dec_input,l_states,enc1,mask1,enc2,mask2)
            dec_input = torch.argmax(tensor,-1)[-1:]
            for i in range(len(lengths1)):
                if preds[i][-1]!=config.END_TOKEN:
                    preds[i].append(int(dec_input[0,i]))
        preds = [x[1:-1] if x[-1]==config.END_TOKEN else x[1:] for x in preds]
        return preds

class Model(nn.Module):
    def __init__(self,config,encoder1,encoder2):
        super(Model,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = Decoder(config)
        self.optimizer = optim.Adam(self.parameters(),lr=config.LR)
        self.to(self.device)
    
    def forward(self,inputs,mode,targets=None):
        if mode:
            return self.train_on_batch(inputs,targets)
        else:
            return self.translate(inputs)

    def train_on_batch(self,inputs,targets):
        self.optimizer.zero_grad()
        self.train()
        enc1, (h1,c1) = self.encoder1(inputs[0])
        enc2, (h2,c2) = self.encoder2(inputs[1])
        h = h1 + h2
        c = c1 + c2
        loss =self.decoder.get_loss(enc1,enc2,(h,c),targets)
        loss.backward()
        self.optimizer.step()
        return float(loss)
    
    def translate(self, inputs):
        with torch.no_grad():
            self.eval()
            enc1, (h1,c1) = self.encoder1(inputs[0])
            enc2, (h2,c2) = self.encoder2(inputs[1])
            h = h1 + h2
            c = c1 + c2
            return self.decoder.translate(enc1,enc2,(h,c))       

    def save(self,path,name):
        torch.save(self.state_dict(),path)

    def load(self,path,name):
        self.load_state_dict(torch.load(path))
