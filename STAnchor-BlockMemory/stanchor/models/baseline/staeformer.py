from __future__ import annotations
import torch
from torch import nn

class _Block(nn.Module):
    def __init__(self,dim,heads,ff,dropout):
        super().__init__(); self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True); self.n1=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,ff),nn.GELU(),nn.Dropout(dropout),nn.Linear(ff,dim)); self.n2=nn.LayerNorm(dim)
    def forward(self,x):
        y,_=self.attn(x,x,x,need_weights=False); return self.n2(self.n1(x+y)+self.ff(self.n1(x+y)))

class STAEformerForecastBackbone(nn.Module):
    def __init__(self,context_length,horizon,num_nodes,input_channels,output_channels,hidden_dim=64,heads=4,layers=2,ff_dim=128,dropout=0.1):
        super().__init__();
        if hidden_dim%heads: raise ValueError("hidden_dim must be divisible by heads")
        self.context_length=context_length; self.horizon=horizon; self.nodes=num_nodes; self.output_channels=output_channels
        self.input_proj=nn.Linear(input_channels,hidden_dim); self.node_emb=nn.Parameter(torch.randn(num_nodes,hidden_dim)*.02); self.time_emb=nn.Parameter(torch.randn(context_length,hidden_dim)*.02)
        self.temporal=nn.ModuleList([_Block(hidden_dim,heads,ff_dim,dropout) for _ in range(layers)]); self.spatial=nn.ModuleList([_Block(hidden_dim,heads,ff_dim,dropout) for _ in range(layers)]); self.output=nn.Linear(context_length*hidden_dim,horizon*output_channels)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=self.context_length or x.shape[2]!=self.nodes: raise ValueError("STAEformer expects [B,T,N,C]")
        b,t,n,_=x.shape; z=self.input_proj(x)+self.time_emb[:t].view(1,t,1,-1)+self.node_emb.view(1,1,n,-1)
        for block in self.temporal: z=block(z.permute(0,2,1,3).reshape(b*n,t,-1)).reshape(b,n,t,-1).permute(0,2,1,3)
        for block in self.spatial: z=block(z.reshape(b*t,n,-1)).reshape(b,t,n,-1)
        z=z.permute(0,2,1,3).reshape(b,n,t*z.shape[-1]); return self.output(z).view(b,n,self.horizon,self.output_channels).transpose(1,2).contiguous()
