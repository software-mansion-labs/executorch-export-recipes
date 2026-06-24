import torch
from blazeface import BlazeFace
from executorch.exir import to_edge

def main():
    # Instantiate front model
    print("Loading BlazeFace model...")
    model = BlazeFace(back_model=False)
    model.load_weights("blazeface.pth")
    model.eval()

    # Create dummy input of shape [1, 3, 128, 128]
    dummy_input = torch.randn(1, 3, 128, 128)

    # Run eager model
    print("Running eager model inference...")
    with torch.no_grad():
        eager_out = model(dummy_input)
    print(f"Eager output shape: regressors={eager_out[0].shape}, classifiers={eager_out[1].shape}")

    # Export using torch.export
    print("Exporting model using torch.export...")
    # Since forward returns a list, and it's a simple model, we can export it.
    # We set strict=True or strict=False.
    with torch.no_grad():
        exported = torch.export.export(model, (dummy_input,), strict=True)
    print("Export successful!")

    # Lower to edge dialect
    print("Lowering to Edge dialect...")
    edge = to_edge(exported)
    print("Edge conversion successful!")

    # Convert to executorch and write to file
    print("Converting to ExecuTorch (.pte)...")
    exec_prog = edge.to_executorch()
    pte_filename = "blazeface.pte"
    with open(pte_filename, "wb") as f:
        f.write(exec_prog.buffer)
    print(f"Model exported successfully to {pte_filename}")

if __name__ == "__main__":
    main()
