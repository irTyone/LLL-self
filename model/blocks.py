
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import generate_mask
import math


class Transformer(nn.Module):
    def __init__(self,config:dict) -> None:
        super().__init__()
        if config.get("emd_dim")%config.get("heads")!=0:
          raise ValueError("注意力头数必须被嵌入维度整除")
        dim=config.get("emd_dim")
        self.heads=config.get("heads")
        self.wq=nn.Linear(dim,dim)
        self.wk=nn.Linear(dim,dim)
        self.wv=nn.Linear(dim,dim)
        self.dropout=nn.Dropout(config.get("drop_rate"))
    
    def forward(self,x):
        batch,content_length,emd=x.shape
        query=self.wq(x)
        key=self.wk(x)
        value=self.wv(x)
        q=query.view(batch,content_length,self.heads,emd//self.heads).transpose(1,2)
        k=key.view(batch,content_length,self.heads,emd//self.heads).transpose(1,2)
        v=value.view(batch,content_length,self.heads,emd//self.heads).transpose(1,2)
        att=q@k.transpose(-2,-1)/math.sqrt(emd//self.heads)
        att=generate_mask(att)
        att_score=F.softmax(att,dim=-1)
        att_score=self.dropout(att_score)
        out=att_score@v
        return out.transpose(1,2).contiguous().view(batch,content_length,emd)
                
class NormLayer(nn.Module):
    def __init__(self, config:dict,eps:float=1e-5) -> None:
        super().__init__()
        self.eps=eps
        self.scale=nn.Parameter(torch.ones(config.get("emd_dim")))
        self.shift=nn.Parameter(torch.zeros(config.get("emd_dim")))
        
    def forward(self,x):          
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x=(x-mean)/(torch.sqrt(var+self.eps))
        return self.scale*norm_x+self.shift                                                                                                                                                                                                                                     
 

class FCLayer(nn.Module):
    def __init__(self,config,extend_d=4 ) -> None:
        super().__init__()
        dim=config.get("emd_dim")
        self.fc1=nn.Sequential(
        nn.Linear(dim,dim*extend_d),
        nn.GELU(),
        nn.Linear(dim*extend_d,dim),
        nn.Dropout(config.get("drop_rate"))
       )


    def forward(self,x):
        return self.fc1(x)
         
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, dim)

    def forward(self, x):
        seq_len = x.size(1) 
        pe_slice = self.pe[:, :seq_len, :] 
        
        
        # print(f"x shape: {x.shape}, pe_slice shape: {pe_slice.shape}") 
        return x + pe_slice




