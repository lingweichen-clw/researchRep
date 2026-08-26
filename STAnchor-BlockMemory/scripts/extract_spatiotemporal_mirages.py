from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def quant(x,p): return float(np.nanquantile(x,p))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--bank',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--num-events',type=int,default=5000); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); rng=np.random.default_rng(a.seed); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    s=pd.HDFStore(a.data,'r'); values=s.get('/df').to_numpy(dtype=np.float32); s.close()
    bank=Path(a.bank); sid0=np.load(bank/'sample_id.npy').astype(np.int64)
    future=np.load(bank/'future_values.npy',mmap_mode='r').astype(np.float32)
    keys=np.load(bank/'node_keys.npy',mmap_mode='r').astype(np.float32)
    E,N,D=keys.shape; H=future.shape[1]
    take=np.arange(E) if E<=a.num_events else np.sort(rng.choice(E,a.num_events,replace=False))
    sid=sid0[take]; ok=(sid>=12)&(sid<len(values)); take=take[ok]; sid=sid0[take]
    X=np.stack([values[x-11:x+1].T for x in sid],axis=0)
    med=np.nanmedian(values,axis=0); X=np.where(np.isfinite(X),X,med[None,:,None])
    Y=np.asarray(future[take,:,:,0]).transpose(0,2,1)
    records=[]
    for n in range(N):
        x=X[:,n,:]; y=Y[:,n,:]; k=np.asarray(keys[take,n,:])
        xz=(x-np.nanmedian(x,0))/(np.nanstd(x,0)+1e-4); yz=(y-np.nanmedian(y,0))/(np.nanstd(y,0)+1e-4)
        nn=NearestNeighbors(n_neighbors=min(25,len(xz))).fit(xz); _, inds=nn.kneighbors(xz)
        for i in range(len(xz)):
            for rnk in range(1,inds.shape[1]):
                j=int(inds[i,rnk])
                if i>=j: continue
                records.append({'node':n,'i':i,'j':j,'context_distance':float(np.linalg.norm(xz[i]-xz[j])/np.sqrt(12)),'future_distance':float(np.linalg.norm(yz[i]-yz[j])/np.sqrt(H)),'key_distance':float(np.linalg.norm(k[i]-k[j])/np.sqrt(D))})
    for n in range(N):
        x=X[:,n,:]; y=Y[:,n,:]; k=np.asarray(keys[take,n,:]); xz=(x-np.nanmedian(x,0))/(np.nanstd(x,0)+1e-4); yz=(y-np.nanmedian(y,0))/(np.nanstd(y,0)+1e-4)
        m=min(500,len(xz)*2); ii=rng.integers(0,len(xz),m); jj=rng.integers(0,len(xz),m)
        for i,j in zip(ii[ii!=jj],jj[ii!=jj]):
            if i>j:i,j=j,i
            records.append({'node':n,'i':int(i),'j':int(j),'context_distance':float(np.linalg.norm(xz[i]-xz[j])/np.sqrt(12)),'future_distance':float(np.linalg.norm(yz[i]-yz[j])/np.sqrt(H)),'key_distance':float(np.linalg.norm(k[i]-k[j])/np.sqrt(D))})
    ctx=np.array([r['context_distance'] for r in records]); fut=np.array([r['future_distance'] for r in records])
    alo,ahi, flo,fhi=quant(ctx,.08),quant(ctx,.92),quant(fut,.08),quant(fut,.92)
    kd=np.array([r['key_distance'] for r in records]); klo,khi=quant(kd,.08),quant(kd,.92)
    A=[r for r in records if r['context_distance']<=alo and r['future_distance']>=fhi and r['key_distance']>=khi]; B=[r for r in records if r['context_distance']>=ahi and r['future_distance']<=flo and r['key_distance']<=klo]
    A.sort(key=lambda r:(-r['future_distance'],-r['key_distance'],r['node'],r['i'],r['j'])); B.sort(key=lambda r:(-r['context_distance'],-r['key_distance'],r['node'],r['i'],r['j']))
    chosen={'context_similar_future_different':A[:3],'context_different_future_similar':B[:3]}
    for typ,rows in chosen.items():
        for r in rows:r.update({'sample_i':int(sid[r['i']]),'sample_j':int(sid[r['j']]),'node_id':r['node']})
    for typ,rows in chosen.items():
        fig,ax=plt.subplots(len(rows),3,figsize=(14,4*max(1,len(rows))),squeeze=False)
        for row,r in enumerate(rows):
            n,i,j=r['node'],r['i'],r['j']
            ax[row,0].plot(X[i,n],marker='o',label=f"event {sid[i]}"); ax[row,0].plot(X[j,n],marker='s',label=f"event {sid[j]}"); ax[row,0].set_title(f"Context d={r['context_distance']:.3f}"); ax[row,0].legend(fontsize=8)
            ax[row,1].plot(Y[i,n],marker='o',label=f"event {sid[i]}"); ax[row,1].plot(Y[j,n],marker='s',label=f"event {sid[j]}"); ax[row,1].set_title(f"Future d={r['future_distance']:.3f}"); ax[row,1].legend(fontsize=8)
            kk=np.asarray(keys[take,n,:]); p=PCA(2,random_state=42).fit(kk); z=p.transform(kk); ax[row,2].scatter(z[:,0],z[:,1],s=8,alpha=.18,color='#777'); hi=p.transform(np.asarray(keys[take[[i,j],],n,:])); ax[row,2].scatter(hi[:,0],hi[:,1],s=55,c=['#d62728','#1f77b4'],edgecolor='black'); ax[row,2].set_title(f"Key PCA d={r['key_distance']:.3f}")
        fig.suptitle(typ.replace('_',' ')); fig.tight_layout(); fig.savefig(out/f'{typ}.png',dpi=180); plt.close(fig)
    payload={'schema_version':1,'selection':{'context_similar_future_different':'context <= P8, future >= P92, key >= P92; sort future then key distance','context_different_future_similar':'context >= P92, future <= P8, key <= P8; sort context then key distance','manual_selection':False},'num_events_used':int(len(take)),'nodes':N,'retrieval_dim':D,'history_steps':12,'horizon':H,'thresholds':{'context_low':alo,'context_high':ahi,'future_low':flo,'future_high':fhi,'key_low':klo,'key_high':khi},'cases':chosen}
    (out/'mirage_cases.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); pd.DataFrame([{'type':t,**r} for t,rows in chosen.items() for r in rows]).to_csv(out/'mirage_cases.csv',index=False); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__':main()


