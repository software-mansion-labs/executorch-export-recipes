"""Validate 4-bit text quality against upstream without allocating a second initialized model.

The native directory contains prompt.bin (i64 IDs) and prefill-logits.bin (f32).
Run in the isolated transformers 5.20.2 reference environment, after native inference.
"""
import json,sys,gc,argparse
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
# Declared acceptance budget for this 4-bit trunk: cosine >= .98, normalized
# RMSE <= .10, top-10 overlap >= .5, and matching greedy first token.
parser=argparse.ArgumentParser(description="Compare saved native prefill logits with the original fp32 upstream model.")
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--native-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
source=args.source.resolve();fixture=args.native_dir
cfg=AutoConfig.from_pretrained(source,trust_remote_code=True)
cls=get_class_from_dynamic_module('modeling_lfm3_omni.Lfm3OmniForConditionalGeneration',str(source));mod=sys.modules[cls.__module__]
with torch.device('meta'): model=mod.Lfm3Model(cfg.text_config)
model.to_empty(device='cpu');model.rotary_emb=mod.Lfm3Model.__init__.__globals__['Lfm3RotaryEmbedding'](cfg.text_config)
state={}
with safe_open(str(source/'model.safetensors'),framework='pt') as f:
 for k in f.keys():
  if k.startswith('model.language_model.'):state[k.removeprefix('model.language_model.')]=f.get_tensor(k).float()
model.load_state_dict(state,assign=True);del state;gc.collect();model.eval();print('reference loaded',flush=True)
ids=torch.from_numpy(np.fromfile(fixture/'prompt.bin',dtype=np.int64).reshape(1,-1))
with torch.no_grad():
 hidden=model(input_ids=ids,use_cache=False).last_hidden_state[:,-1]
 expected=torch.nn.functional.linear(hidden,model.embed_tokens.weight).float()
 actual=torch.from_numpy(np.fromfile(fixture/'prefill-logits.bin',dtype=np.float32).reshape(1,128000))
 error=(actual-expected).float();cos=torch.nn.functional.cosine_similarity(actual.flatten(),expected.flatten(),dim=0).item();nrmse=(error.square().mean().sqrt()/expected.square().mean().sqrt()).item();overlap=len(set(actual.topk(10).indices[0].tolist())&set(expected.topk(10).indices[0].tolist()))/10
 result={'cosine':cos,'normalized_rmse':nrmse,'max_abs':error.abs().max().item(),'top10_overlap':overlap,'source_first_token':int(expected.argmax()),'native_first_token':int(actual.argmax()),'pass':cos>=.98 and nrmse<=.10 and overlap>=.5 and int(actual.argmax())==int(expected.argmax())}
 print(result,flush=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
 assert result['pass'],result
