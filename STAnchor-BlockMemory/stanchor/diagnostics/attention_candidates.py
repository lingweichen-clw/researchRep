from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig
from stanchor.engine.common import build_data_and_graph, load_checkpoint, load_pretrained_model
from stanchor.engine.target import _validate_bank, build_downstream_model, checkpoint_bank_level_weight, checkpoint_candidate_protocol, checkpoint_downstream_mode, retrieve_for_downstream_mode
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.utils import resolve_device

def _rho(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0: return None
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else None

@torch.no_grad()
def diagnose_attention_checkpoint(config: ExperimentConfig, pretrained_checkpoint: str | Path, downstream_checkpoint: str | Path, bank_path: str | Path, split='val', max_batches=None) -> dict[str, Any]:
    device=resolve_device(config.runtime.device); data, graph_cpu=build_data_and_graph(config); graph=graph_cpu.to(device)
    pretrained,_=load_pretrained_model(config,pretrained_checkpoint,data.series.slots_per_day,device); checkpoint=load_checkpoint(downstream_checkpoint,device)
    mode=checkpoint_downstream_mode(checkpoint); protocol=checkpoint_candidate_protocol(checkpoint)
    config=replace(config,bank=replace(config.bank,level_weight=checkpoint_bank_level_weight(checkpoint,config.bank.level_weight)),target=replace(config.target,downstream_mode=mode,candidate_protocol=protocol))
    downstream=build_downstream_model(config,graph).to(device); downstream.load_state_dict(checkpoint['downstream_state_dict'],strict=True); pretrained.eval(); downstream.eval()
    if not hasattr(downstream.error_corrector,'last_attention'): raise RuntimeError('current corrector does not expose validation attention')
    loader=DataLoader(getattr(data,split),batch_size=config.target.batch_size,shuffle=False,num_workers=config.data.num_workers)
    top1=[]; top5=[]; ent=[]; aa=[]; ee=[]; br=[]; batches=0; queries=0
    with MemoryBank(bank_path,expected_schema_version=(2 if pretrained.model_config.profile_dim>0 else 1)) as bank:
        _validate_bank(bank,pretrained,graph_cpu,data.scaler.state_dict())
        if checkpoint.get('bank_manifest')!=bank.manifest.to_dict(): raise ValueError('diagnostic bank differs from the bank used for downstream training')
        retriever=TwoStageRetriever(bank,config.bank.event_top_r,config.bank.node_top_k,config.bank.level_weight,config.bank.level_temperature,config.bank.search_temperature,device)
        for i,batch in enumerate(loader):
            if max_batches is not None and i>=max_batches: break
            x=batch['x'].to(device); ox=batch['x_observed'].to(device); candidates,aggregation=retrieve_for_downstream_mode(mode,pretrained,retriever,bank,data,graph,batch,x,ox,device,candidate_protocol=protocol); downstream(x,candidates,aggregation); attention=downstream.error_corrector.last_attention
            target=data.scaler.inverse_transform_torch(batch['y'].to(device)); mean=torch.as_tensor(data.scaler.mean,dtype=aggregation.candidate_futures.dtype,device=device)[None,None,:,None,:]; std=torch.as_tensor(data.scaler.std,dtype=aggregation.candidate_futures.dtype,device=device)[None,None,:,None,:]; future=aggregation.candidate_futures*(std+data.scaler.eps)+mean; valid=aggregation.candidate_masks.bool().all(-1)&aggregation.valid.bool().all(-1,keepdim=True); errors=(future-target.unsqueeze(3)).abs().mean(-1); valid=valid&torch.isfinite(errors); loc=valid.any(-1)
            if not bool(loc.any()): continue
            a=attention.float()[loc]; e=errors[loc]; v=valid[loc]; k=a.shape[-1]; top1.extend(a[:,0].cpu().tolist()); top5.extend(a[:,:min(5,k)].sum(-1).cpu().tolist()); ent.extend((-(a.clamp_min(1e-12)*a.clamp_min(1e-12).log()).sum(-1)).cpu().tolist())
            for av,ev,vm in zip(a.cpu().numpy(),e.cpu().numpy(),v.cpu().numpy()):
                m=vm.astype(bool)
                if m.sum()<2: continue
                aa.extend(av[m].tolist()); ee.extend(ev[m].tolist()); order=np.argsort(-av); br.append(int(np.flatnonzero(order==np.argmin(np.where(m,ev,np.inf)))[0])+1)
            batches+=1; queries+=int(x.shape[0])
    return {'schema_version':1,'diagnostic':'validation_only_candidate_attention','split':split,'queries':queries,'batches':batches,'candidate_protocol':protocol,'node_top_k':config.bank.node_top_k,'retrieval_rank1_attention_mean':float(np.mean(top1)),'retrieval_top5_cumulative_attention_mean':float(np.mean(top5)),'attention_entropy_mean':float(np.mean(ent)),'attention_entropy_std':float(np.std(ent)),'attention_vs_candidate_error_spearman':_rho(aa,ee),'oracle_best_candidate_attention_rank_mean':float(np.mean(br)) if br else None,'oracle_best_candidate_attention_rank_p50':float(np.median(br)) if br else None,'future_information_boundary':'candidate future and target are used only for validation diagnostics'}