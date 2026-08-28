from __future__ import annotations
import argparse,json,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from stanchor.config import load_config,resolve_project_path
from stanchor.diagnostics.attention_candidates import diagnose_attention_checkpoint
from stanchor.utils import save_json
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--pretrained-checkpoint',required=True); p.add_argument('--downstream-checkpoint',required=True); p.add_argument('--bank',required=True); p.add_argument('--split',default='val'); p.add_argument('--output',required=True); p.add_argument('--max-batches',type=int,default=None); a=p.parse_args()
r=diagnose_attention_checkpoint(load_config(a.config),a.pretrained_checkpoint,a.downstream_checkpoint,a.bank,a.split,a.max_batches); o=resolve_project_path(a.output); save_json(o,r); print(json.dumps(r,ensure_ascii=False,indent=2)); print(f'diagnostic output: {o}')