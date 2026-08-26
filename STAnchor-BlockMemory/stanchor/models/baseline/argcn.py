from __future__ import annotations
import torch
from torch import nn

class _AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, support, cheb_k=2):
        super().__init__(); self.register_buffer("support", support.float()); self.cheb_k=cheb_k
        self.weight=nn.Parameter(torch.empty(support.shape[0]*cheb_k*dim_in, dim_out)); self.bias=nn.Parameter(torch.zeros(dim_out)); nn.init.xavier_uniform_(self.weight)
    def forward(self,x):
        mats=[]
        for a in self.support:
            mats.append(torch.eye(a.shape[0],device=x.device,dtype=x.dtype)); mats.append(a.to(dtype=x.dtype))
        feats=[torch.einsum("nm,bmc->bnc",a,x) for a in mats[:self.support.shape[0]*self.cheb_k]]
        return torch.einsum("bni,io->bno",torch.cat(feats,-1),self.weight.to(x.dtype))+self.bias.to(x.dtype)

class _Cell(nn.Module):
    def __init__(self,nodes,dim_in,hidden,support):
        super().__init__(); self.hidden=hidden; self.gate=_AGCN(dim_in+hidden,2*hidden,support); self.update=_AGCN(dim_in+hidden,hidden,support)
    def forward(self,x,state):
        z,r=torch.sigmoid(self.gate(torch.cat((x,state),-1))).chunk(2,-1); h=torch.tanh(self.update(torch.cat((x,z*state),-1))); return r*state+(1-r)*h

class ARGCNForecastBackbone(nn.Module):
    def __init__(self,context_length,horizon,num_nodes,input_channels,output_channels,graph_support,hidden_dim=64,num_layers=2):
        super().__init__(); self.nodes=num_nodes; self.horizon=horizon; self.output_channels=output_channels
        self.cells=nn.ModuleList([_Cell(num_nodes,input_channels if i==0 else hidden_dim,hidden_dim,graph_support) for i in range(num_layers)]); self.proj=nn.Linear(hidden_dim,horizon*output_channels)
    def forward(self,x):
        if x.ndim!=4 or x.shape[2]!=self.nodes: raise ValueError("ARGCN expects [B,T,N,C]")
        states=[x.new_zeros(x.shape[0],self.nodes,c.hidden) for c in self.cells]
        for t in range(x.shape[1]):
            value=x[:,t]
            for i,cell in enumerate(self.cells): states[i]=cell(value,states[i]); value=states[i]
        return self.proj(states[-1]).view(x.shape[0],self.nodes,self.horizon,self.output_channels).transpose(1,2).contiguous()
