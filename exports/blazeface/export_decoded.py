import torch
import torch.nn as nn
import numpy as np
from blazeface import BlazeFace
from executorch.exir import to_edge
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer

class BlazeFaceDecodedWrapper(nn.Module):
    def __init__(self, eager_model, anchors_npy_path):
        super().__init__()
        self.model = eager_model
        
        # Load anchors as numpy array and register as a buffer
        anchors = np.load(anchors_npy_path)
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        
        self.x_scale = eager_model.x_scale
        self.y_scale = eager_model.y_scale
        self.w_scale = eager_model.w_scale
        self.h_scale = eager_model.h_scale

    def forward(self, x):
        raw_regs, raw_clss = self.model(x)
        
        # raw_regs shape: [B, 896, 16]
        # self.anchors shape: [896, 4]
        # anchors layout: [x_center, y_center, w, h]
        
        dx = raw_regs[..., 0]
        dy = raw_regs[..., 1]
        dw = raw_regs[..., 2]
        dh = raw_regs[..., 3]
        
        anchor_x = self.anchors[:, 0]
        anchor_y = self.anchors[:, 1]
        anchor_w = self.anchors[:, 2]
        anchor_h = self.anchors[:, 3]
        
        # Decode directly to model pixel coordinates
        x_center = dx * anchor_w + anchor_x * self.x_scale
        y_center = dy * anchor_h + anchor_y * self.y_scale
        w = dw * anchor_w
        h = dh * anchor_h
        
        ymin = y_center - h / 2.0
        xmin = x_center - w / 2.0
        ymax = y_center + h / 2.0
        xmax = x_center + w / 2.0
        
        # Output standard [xmin, ymin, xmax, ymax] order (xyxy format)
        decoded_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        
        kp_list = []
        for k in range(6):
            offset = 4 + k * 2
            kp_x = raw_regs[..., offset] * anchor_w + anchor_x * self.x_scale
            kp_y = raw_regs[..., offset + 1] * anchor_h + anchor_y * self.y_scale
            kp_conf = torch.ones_like(kp_x)
            kp_list.append(torch.stack([kp_x, kp_y, kp_conf], dim=-1))
            
        decoded_keypoints = torch.stack(kp_list, dim=-2)
        return decoded_boxes.squeeze(0), torch.sigmoid(raw_clss).squeeze(0).squeeze(-1), decoded_keypoints.squeeze(0)

def main():
    print("Loading eager BlazeFace back model...")
    eager_model = BlazeFace(back_model=True)
    eager_model.load_weights("blazefaceback.pth")
    eager_model.eval()

    print("Wrapping model with anchor decoding math...")
    wrapped_model = BlazeFaceDecodedWrapper(eager_model, "anchorsback.npy")
    wrapped_model.eval()

    # Create dummy input of shape [1, 3, 256, 256] for back model
    dummy_input = torch.randn(1, 3, 256, 256)

    # Test run eagerly
    with torch.no_grad():
        dec_boxes, raw_clss, dec_kps = wrapped_model(dummy_input)
        print(f"Eager test successful! Outputs: boxes={dec_boxes.shape}, clss={raw_clss.shape}, kps={dec_kps.shape}")

    # Export using torch.export
    print("Exporting model using torch.export...")
    with torch.no_grad():
        exported = torch.export.export(wrapped_model, (dummy_input,), strict=True)
    print("Export successful!")

    # Lower to edge dialect
    print("Lowering to Edge dialect...")
    edge = to_edge(exported)
    print("Edge conversion successful!")

    # Lower to XNNPACK backend delegate
    print("Lowering to XNNPACK backend delegate...")
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    edge = edge.to_backend(XnnpackPartitioner())

    # Convert to executorch
    print("Converting to ExecuTorch (.pte)...")
    exec_prog = edge.to_executorch()
    
    # Save to file
    pte_filename = "blazeface_decoded.pte"
    with open(pte_filename, "wb") as f:
        f.write(exec_prog.buffer)
    print(f"Model exported successfully to {pte_filename}!")

    # Load and test ExecuTorch model output parity
    print("Verifying parity with ExecuTorch model...")
    et_model = _load_for_executorch_from_buffer(exec_prog.buffer)
    et_outputs = et_model.run_method("forward", (dummy_input,))
    et_box, et_cls, et_kp = et_outputs[0], et_outputs[1], et_outputs[2]

    diff_box = torch.max(torch.abs(dec_boxes - et_box)).item()
    diff_cls = torch.max(torch.abs(raw_clss - et_cls)).item()
    diff_kp = torch.max(torch.abs(dec_kps - et_kp)).item()

    print(f"Max difference in boxes: {diff_box:.2e}")
    print(f"Max difference in classifiers: {diff_cls:.2e}")
    print(f"Max difference in keypoints: {diff_kp:.2e}")
    if diff_box < 1e-4 and diff_cls < 1e-4 and diff_kp < 1e-4:
        print("SUCCESS: Numeric parity validated!")
    else:
        print("WARNING: Large numerical discrepancy!")

if __name__ == "__main__":
    main()
