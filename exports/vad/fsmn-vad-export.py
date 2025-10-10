import torch

from vad_model import VAD

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from torch.export import Dim, export

trained_model = VAD(
    input_dim=400,
    input_affine_dim=140,
    fsmn_layers=4,
    linear_dim=250,
    proj_dim=128,
    lorder=20,
    rorder=0,
    lstride=1,
    rstride=1,
    output_affine_dim=140,
    output_dim=248,
    use_softmax=True
)

original_state_dict = torch.load("models/model.pt", map_location="cpu")

prefix = "encoder."
corrected_state_dict = {
    k[len(prefix):]: v
    for k, v in original_state_dict.items()
    if k.startswith(prefix)
}

trained_model.load_state_dict(corrected_state_dict)
trained_model.eval() 

dynamic_shapes = {
    "input" : {
        0: Dim("T", min=100, max =1000) # input audio 1s - 10s
    }
}
dummy_feats = (torch.randn(100, 512,dtype=torch.float32),)

with torch.no_grad():
  exported_program = export(trained_model, dummy_feats,dynamic_shapes=dynamic_shapes)

  edge_compile_config = EdgeCompileConfig(_check_ir_validity=False)
  et_program = to_edge_transform_and_lower(
      exported_program,
      partitioner=[XnnpackPartitioner()],
      compile_config=edge_compile_config
  ).to_executorch()

  with open("fsmn-vad-dynamic-6.pte", "wb") as file:
      et_program.write_to_file(file)
