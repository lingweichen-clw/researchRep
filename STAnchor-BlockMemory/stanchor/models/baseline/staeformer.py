from __future__ import annotations
import torch
from torch import nn

class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__(); self.model_dim=model_dim; self.num_heads=num_heads; self.head_dim=model_dim//num_heads; self.mask=mask; self.q=nn.Linear(model_dim,model_dim); self.k=nn.Linear(model_dim,model_dim); self.v=nn.Linear(model_dim,model_dim); self.out=nn.Linear(model_dim,model_dim)
    def forward(self,q,k,v):
        b=q.shape[0]; tl=q.shape[-2]; sl=k.shape[-2]; q=self.q(q); k=self.k(k); v=self.v(v); q=torch.cat(torch.split(q,self.head_dim,dim=-1),dim=0); k=torch.cat(torch.split(k,self.head_dim,dim=-1),dim=0); v=torch.cat(torch.split(v,self.head_dim,dim=-1),dim=0); score=(q@k.transpose(-1,-2))/(self.head_dim**0.5)
        if self.mask: score=score.masked_fill(~torch.ones(tl,sl,device=q.device,dtype=torch.bool).tril(),-torch.inf)
        out=torch.softmax(score,dim=-1)@v; out=torch.cat(torch.split(out,b,dim=0),dim=-1); return self.out(out)

class SelfAttentionLayer(nn.Module):
    def __init__(self,model_dim,feed_forward_dim=2048,num_heads=8,dropout=0,mask=False):
        super().__init__(); self.attn=AttentionLayer(model_dim,num_heads,mask); self.ff=nn.Sequential(nn.Linear(model_dim,feed_forward_dim),nn.ReLU(inplace=True),nn.Linear(feed_forward_dim,model_dim)); self.ln1=nn.LayerNorm(model_dim); self.ln2=nn.LayerNorm(model_dim); self.d1=nn.Dropout(dropout); self.d2=nn.Dropout(dropout)
    def forward(self,x,dim=-2):
        x=x.transpose(dim,-2); x=self.ln1(x+self.d1(self.attn(x,x,x))); x=self.ln2(x+self.d2(self.ff(x))); return x.transpose(dim,-2)

class STAEformerForecastBackbone(nn.Module):
    """STAEformer with the source default dimensions and mixed projection."""
    def __init__(self,context_length,horizon,num_nodes,input_channels,output_channels,steps_per_day=288,input_embedding_dim=24,tod_embedding_dim=24,dow_embedding_dim=24,spatial_embedding_dim=0,adaptive_embedding_dim=80,feed_forward_dim=256,heads=4,layers=3,dropout=0.1):
        super().__init__(); self.context_length=context_length; self.horizon=horizon; self.nodes=num_nodes; self.input_dim=input_channels; self.output_channels=output_channels; self.steps_per_day=steps_per_day; self.model_dim=input_embedding_dim+tod_embedding_dim+dow_embedding_dim+spatial_embedding_dim+adaptive_embedding_dim
        self.input_proj=nn.Linear(input_channels,input_embedding_dim); self.tod=nn.Embedding(steps_per_day,tod_embedding_dim); self.dow=nn.Embedding(7,dow_embedding_dim); self.adaptive=nn.Parameter(torch.empty(context_length,num_nodes,adaptive_embedding_dim)); nn.init.xavier_uniform_(self.adaptive)
        self.temporal=nn.ModuleList([SelfAttentionLayer(self.model_dim,feed_forward_dim,heads,dropout) for _ in range(layers)]); self.spatial=nn.ModuleList([SelfAttentionLayer(self.model_dim,feed_forward_dim,heads,dropout) for _ in range(layers)]); self.output_proj=nn.Linear(context_length*self.model_dim,horizon*output_channels)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=self.context_length or x.shape[2]!=self.nodes: raise ValueError("STAEformer expects [B,T,N,C]")
        b,t,n,_=x.shape; # Mainline has no covariate channels; use deterministic slot ids and weekday zero.
        tod_idx=torch.arange(t,device=x.device).view(1,t,1).expand(b,t,n)%self.steps_per_day; dow_idx=torch.zeros((b,t,n),device=x.device,dtype=torch.long)
        z=torch.cat((self.input_proj(x),self.tod(tod_idx),self.dow(dow_idx),self.adaptive[:t].unsqueeze(0).expand(b,-1,-1,-1)),dim=-1)
        for layer in self.temporal: z=layer(z,dim=1)
        for layer in self.spatial: z=layer(z,dim=2)
        z=z.transpose(1,2).reshape(b,n,t*self.model_dim); return self.output_proj(z).view(b,n,self.horizon,self.output_channels).transpose(1,2).contiguous()
