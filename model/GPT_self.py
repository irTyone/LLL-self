from re import X
import torch.nn as nn
import torch
import torch.functional as F

from model.blocks import Transformer,NormLayer,FCLayer,PositionalEncoding

class LLGPT(nn.Module):
    def __init__(self,config:dict,block_nums:int,is_transformer_block:int=0) -> None:
        super().__init__()
        self.transformer_block_num=is_transformer_block
        self.embedding=nn.Embedding(config.get("vocab_size"),config.get("emd_dim"))
        print(self.embedding.weight.shape)
        self.posembeding=PositionalEncoding(config.get("emd_dim"),max_len=config.get("vocab_size"))
        self.base_trf=nn.ModuleList()
        self.base_fc=nn.ModuleList()
        for _ in range(block_nums):
            self.base_trf.append(nn.Sequential(
            NormLayer(config),
            Transformer(config)
            ))
        for _ in range(block_nums):
            self.base_fc.append(nn.Sequential(
            NormLayer(config),
            FCLayer(config)
            ))
        if is_transformer_block!=0:
            self.transformer_block=nn.ModuleList([Transformer(config) for _ in range(is_transformer_block)])
        self.final_norm=NormLayer(config)
        self.logits=nn.Linear(config.get("emd_dim"),config.get("vocab_size"))

    def forward(self,x):
        hidden_value=self.embedding(x)
        hidden_value=self.posembeding(hidden_value)
        for trf,fc in zip(self.base_trf,self.base_fc):
            hidden_value=hidden_value+trf(hidden_value)
            hidden_value=hidden_value+fc(hidden_value)
        if self.transformer_block_num!=0:
            for trf_blc in self.transformer_block:
                hidden_value=hidden_value+trf_blc(hidden_value)
        final_norm=self.final_norm(hidden_value)
        out_logits=self.logits(final_norm)
        return out_logits

        
           
