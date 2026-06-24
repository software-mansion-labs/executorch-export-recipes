import torch
import numpy as np
from blazeface import BlazeFace
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer

def main():
    # Load eager model
    print("Loading eager model...")
    eager_model = BlazeFace(back_model=False)
    eager_model.load_weights("blazeface.pth")
    eager_model.eval()

    # Load ExecuTorch model
    print("Loading ExecuTorch model...")
    with open("blazeface.pte", "rb") as f:
        pte_buffer = f.read()
    et_model = _load_for_executorch_from_buffer(pte_buffer)

    # Generate a constant or random input tensor
    # We can use values in range [-1, 1] or [0, 255]
    # In our model, forward expects normalized input, or raw image that isn't preprocessed yet?
    # Wait, BlazeFace.forward does:
    # x = F.pad(x, (1, 2, 1, 2), "constant", 0)
    # So the model's actual forward method expects the input to be preprocessed (e.g., float, typically in [-1, 1]).
    input_data = torch.randn(1, 3, 128, 128)

    # Run eager model
    with torch.no_grad():
        eager_reg, eager_cls = eager_model(input_data)

    # Run ExecuTorch model
    # Note: run_method takes a tuple of inputs and returns a list/tuple of outputs.
    # The input to run_method must be a PyTorch tensor (or similar).
    et_outputs = et_model.run_method("forward", (input_data,))
    et_reg, et_cls = et_outputs[0], et_outputs[1]

    # Verify shapes
    print(f"Eager Regressor Shape: {eager_reg.shape}, Classifiers Shape: {eager_cls.shape}")
    print(f"ExecuTorch Regressor Shape: {et_reg.shape}, Classifiers Shape: {et_cls.shape}")

    # Verify values
    diff_reg = torch.max(torch.abs(eager_reg - et_reg)).item()
    diff_cls = torch.max(torch.abs(eager_cls - et_cls)).item()

    print(f"Max difference in regressors: {diff_reg:.2e}")
    print(f"Max difference in classifiers: {diff_cls:.2e}")

    # Assert correctness
    if diff_reg < 1e-4 and diff_cls < 1e-4:
        print("SUCCESS: ExecuTorch model output matches PyTorch eager model output!")
    else:
        print("FAILURE: Large discrepancy between ExecuTorch and eager model!")

if __name__ == "__main__":
    main()
