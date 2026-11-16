"""Isolate XNNPACK vs PT2E dynamic activation granularity on one linear."""
import argparse
import json
from pathlib import Path
import torch
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e,convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer,get_symmetric_quantization_config
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', required=True)
args=parser.parse_args()
root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(17)
class Linear(torch.nn.Module):
 def __init__(self):super().__init__();self.linear=torch.nn.Linear(1024,1024)
 def forward(self,x):return self.linear(x)
rows=[]
with torch.inference_mode():
 for frames in [1,3,63]:
  inputs=(torch.randn(1,frames,1024)*torch.linspace(.1,3,frames).reshape(1,-1,1),)
  module=Linear().eval()
  p=prepare_pt2e(torch.export.export(module,inputs).module(check_guards=False),XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True,is_dynamic=True)));p(*inputs);q=convert_pt2e(p)
  expected=q(*inputs)
  # Evaluating each token separately forces PT2E's per-tensor chooser to
  # perform per-token activation quantization without changing weight scales.
  per_token=torch.cat([q.forward(inputs[0][:,i:i+1,:]) for i in range(frames)],dim=1)
  program=to_edge_transform_and_lower(torch.export.export(q,inputs),partitioner=[XnnpackPartitioner()]).to_executorch()
  path=root/f'diagnostic-linear-{frames}.pte'
  with path.open('wb') as f:program.write_to_file(f)
  runtime=Runtime.get().load_program(str(path));actual=runtime.load_method('forward').execute([inputs[0].contiguous()])[0].clone()
  row=dict(frames=frames,full_tensor_vs_runtime_max=float((expected-actual).abs().max()),per_token_vs_runtime_max=float((per_token-actual).abs().max()),full_tensor_outside_1e3=int((~torch.isclose(expected,actual,atol=1e-3,rtol=1e-3)).sum()),per_token_outside_1e3=int((~torch.isclose(per_token,actual,atol=1e-3,rtol=1e-3)).sum()),quant_chooser_ops=sorted({str(n.target) for n in q.graph.nodes if 'choose_qparams' in str(n.target)}))
  rows.append(row);print(row,flush=True)
(root/'dynamic-quant-diagnostic.json').write_text(json.dumps(rows,indent=2))
