"""Export mixed precision Fluid: dynamic INT8 encoder linears, FP32 elsewhere."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root/'xnnpack/config.json').read_text())
    config['quantize_encoder'] = 'int8dyn'
    for variant in config['variants']:
        if '_encoder_' in variant['file']:
            variant.update(file=variant['file'].replace('_fp32.pte', '_int8dyn.pte'),
                           precision='int8dyn', quantized=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/'export-config.json').write_text(json.dumps(config, indent=2))
    spec = importlib.util.spec_from_file_location('fluid_quantized_export', root/'xnnpack/export.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.export(config, str(output))


if __name__ == '__main__':
    main()
