"""Vendored PyTorch port of Supertonic 3 (from Supertone/supertonic-3 ONNX).

Weight extraction (convert.load_models) + the four sub-model definitions
(models/modules) used by the ExecuTorch export. pipeline.py is the host-side
reference (tokenization + flow-matching loop) and is not used by export.py.
"""
