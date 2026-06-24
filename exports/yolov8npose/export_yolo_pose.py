#!/usr/bin/env python3
import argparse
import sys
import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOv8PoseDecodedWrapper(nn.Module):
    """
    Wraps a YOLOv8 Pose model to produce outputs matching the BlazeFace decoded schema,
    optionally filtering to keep only the top K highest confidence predictions.
    
    BlazeFace decoded schema outputs:
    1. boxes: [num_anchors, 4] -> [xmin, ymin, xmax, ymax] in absolute pixel coordinates.
    2. scores: [num_anchors] -> confidence score (0 to 1) for each anchor.
    3. keypoints: [num_anchors, 17, 3] -> [x, y, confidence] for the 17 pose landmarks.
    """
    def __init__(self, eager_model, topk=None):
        super().__init__()
        self.model = eager_model
        self.topk = topk
        
    def forward(self, x):
        # Run raw YOLOv8 model forward pass
        outputs = self.model(x)
        preds = outputs[0] # Shape: [B, 56, num_anchors]
        
        # Squeeze batch dimension to conform to BlazeFace wrapper schema
        preds = preds.squeeze(0) # Shape: [56, num_anchors]
        num_anchors = preds.shape[1]
        
        # Extract bounding box center coords and dimensions
        cx = preds[0, :]
        cy = preds[1, :]
        w = preds[2, :]
        h = preds[3, :]
        
        # Convert cx, cy, w, h to xmin, ymin, xmax, ymax
        xmin = cx - w / 2.0
        ymin = cy - h / 2.0
        xmax = cx + w / 2.0
        ymax = cy + h / 2.0
        
        decoded_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1) # Shape: [num_anchors, 4]
        
        # Extract classification scores (already sigmoid-activated in YOLOv8 eval mode)
        scores = preds[4, :] # Shape: [num_anchors]
        
        # Extract and format keypoints [num_anchors, 17, 3]
        # YOLOv8 keypoint format is [17*3, num_anchors] which is [kp1_x, kp1_y, kp1_conf, kp2_x, ...]
        kps_raw = preds[5:, :].permute(1, 0) # Shape: [num_anchors, 51]
        decoded_keypoints = kps_raw.reshape(num_anchors, 17, 3) # Shape: [num_anchors, 17, 3]
        
        # Filter predictions if topk is set
        if self.topk is not None and self.topk > 0:
            topk_scores, topk_indices = torch.topk(scores, self.topk, dim=0, largest=True, sorted=True)
            decoded_boxes = decoded_boxes[topk_indices]
            decoded_keypoints = decoded_keypoints[topk_indices]
            scores = topk_scores
            
        return decoded_boxes, scores, decoded_keypoints

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 Pose model to conform to BlazeFace schema in ExecuTorch format.")
    parser.add_argument("--model-path", type=str, default="yolov8n-pose.pt", help="Path to the PyTorch/Ultralytics model file (.pt)")
    parser.add_argument("--output-path", type=str, default="yolov8n_pose_decoded.pte", help="Target output file path for the .pte model")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size (width and height must be equal)")
    parser.add_argument("--topk", type=int, default=300, help="If > 0, only returns the top K highest scoring detections (default: 300). Pass 0 to return all anchors.")
    parser.add_argument("--no-xnnpack", action="store_true", help="Disable XNNPACK backend delegate optimization")
    args = parser.parse_args()
    
    # Resolve topk
    topk_val = args.topk if args.topk > 0 else None
    
    print(f"Loading YOLOv8-pose model from {args.model_path}...")
    try:
        model = YOLO(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    pyt_model = model.model
    pyt_model.eval()
    
    print(f"Wrapping model to conform to BlazeFace decoded schema (topk={args.topk})...")
    wrapped_model = YOLOv8PoseDecodedWrapper(pyt_model, topk=topk_val)
    wrapped_model.eval()
    
    # Create dummy input based on img-size
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    print(f"Created dummy input of shape {list(dummy_input.shape)}")
    
    # Run eager check
    print("Running eager inference check...")
    with torch.no_grad():
        dec_boxes, raw_clss, dec_kps = wrapped_model(dummy_input)
    print(f"Eager inference successful!")
    print(f"  - decoded_boxes: {list(dec_boxes.shape)}")
    print(f"  - scores:        {list(raw_clss.shape)}")
    print(f"  - keypoints:     {list(dec_kps.shape)}")
    
    # Export using torch.export
    print("Exporting model using torch.export...")
    try:
        from executorch.exir import to_edge
    except ImportError:
        print("Error: ExecuTorch (executorch) library is not installed in the current environment.")
        sys.exit(1)
        
    with torch.no_grad():
        exported = torch.export.export(wrapped_model, (dummy_input,), strict=True)
    print("torch.export successful!")
    
    # Lower to Edge dialect
    print("Lowering to Edge dialect...")
    edge = to_edge(exported)
    print("Edge dialect lowering successful!")
    
    # Lower to XNNPACK delegate if not disabled
    if not args.no_xnnpack:
        print("Lowering to XNNPACK backend delegate...")
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            edge = edge.to_backend(XnnpackPartitioner())
            print("XNNPACK lowering successful!")
        except Exception as e:
            print(f"Warning: XNNPACK lowering failed. Proceeding with standard Edge dialect. Error: {e}")
            
    # Convert to ExecuTorch program
    print("Converting to ExecuTorch format...")
    exec_prog = edge.to_executorch()
    
    # Save target file
    with open(args.output_path, "wb") as f:
        f.write(exec_prog.buffer)
    print(f"SUCCESS: Model exported successfully to {args.output_path}!")
    
    # Load and test ExecuTorch model output parity
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
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
        if diff_box < 5e-3 and diff_cls < 1e-4 and diff_kp < 5e-3:
            print("Parity verification: PASSED")
        else:
            print("Parity verification: WARNING (Larger than expected difference)")
    except Exception as e:
        print(f"Skipping verification check: {e}")

if __name__ == "__main__":
    main()
