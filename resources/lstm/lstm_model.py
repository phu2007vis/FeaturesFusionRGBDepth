import torch 
import torch.nn as nn
import math
from resources.lstm.PAttention import PCrossAttentionModule ,MLP
from resources.utils import *
#use  in future
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000,dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.linspace(0, max_len - 1, steps=max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[0,:x.size(0), :]
        return self.dropout(x)

class LSTMModel(nn.Module):
    def __init__(self,
                 num_classes,
                 n_frames,
                 num_keypoints,
                 hidden_size= 512,
                 numlayers  = 2,
                 num_attention_layers = 2,
                 num_head_attention = 8,
                 dropout = 0.1,
                 activate = "ReLU",
                 *args,
                 **kwargs) -> None:
        super(LSTMModel,self).__init__()
        self.num_layers = numlayers
        self.hidden_size = hidden_size
        self.lstm_time  = nn.LSTM(num_keypoints, hidden_size, numlayers, batch_first=True)
        self.lstm_spatial = nn.LSTM(n_frames, hidden_size, numlayers, batch_first=True)
        self.cross_attention_module = PCrossAttentionModule(hidden_size,num_attention_layers,num_head_attention,dropout,activate)
        self.position_embeding = PositionalEncoding(hidden_size)
        self.mlp  = MLP([hidden_size,num_classes],0)
        
        self.mlp.mlp_main[0].linear_layer_main[1] = nn.Identity()
        self.num_classes = num_classes
    def forward(self,x):
        x_time,x_spatial = x
        h0 = torch.zeros(self.num_layers, x_spatial.shape[0], self.hidden_size).to(x_time.device)
        c0 = torch.zeros(self.num_layers, x_spatial.shape[0], self.hidden_size).to(x_time.device)
        
        # Forward propagate LSTM
        out1, _ = self.lstm_time(x_time, (h0, c0))  
        out2, _ = self.lstm_spatial(x_spatial, (h0,c0))
        time_out = out1[:,-1,:]
        spatial_out = out2[:,-1,:]
   
        self.out_attention = self.cross_attention_module((time_out,spatial_out))
        return self.mlp(self.out_attention)
        

if __name__ == "__main__":
    
    # model = Conv1d(1,1,3,1)
    # sum = 0 
    time = 32
    num_join = 75
    x_time = torch.randn(2,time, num_join)
    x_spatial = x_time.transpose(1,2)
    # possition_embedding = PositionalEncoding(64)
    # possition_embedding(x).shape
    model = LSTMModel(100,time,num_join,512)
    output = model((x_time,x_spatial))
    for param in model.parameters():
        sum+= param.numel()
    
    
    
    