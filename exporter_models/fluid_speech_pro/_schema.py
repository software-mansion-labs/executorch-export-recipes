"""Companion dynamic schemas for RNE's existing low-level Model API."""
import json


def dynamic_schema(component):
    def constant(value):
        return dict(kind='constant', value=value)

    def domain(low, high, step=1):
        return dict(kind='range', range=dict(min=low, max=high, step=step))

    def tensor(dtype, shape):
        return dict(kind='Tensor', dtype=dtype,
                    shape=[constant(x) if isinstance(x, int) else x for x in shape])

    def ref(side, slot, axis):
        return dict(paramSide=side, tensorIdx=slot, dimIdx=axis)

    length = tensor('int64', [1])
    if component == 'encoder':
        frames = domain(3, 126)
        schema = dict(forward=dict(
            inputs=[tensor('float32', [1, frames, 1024]), length],
            outputs=[tensor('float32', [1, 1024, frames]), length],
            runtimeConstraints=[dict(kind='equality', dims=[ref('input', 0, 1), ref('output', 0, 2)])]))
    elif component == 'subsample':
        schema = {}
        for residue in range(8):
            low, high = (17-residue+7)//8, (1001-residue)//8
            offset = int(residue != 0)
            schema[f'forward_{residue}'] = dict(
                inputs=[tensor('float32', [1, 128, domain(8*low+residue, 8*high+residue, 8)]), length],
                outputs=[tensor('float32', [1, domain(low+offset, high+offset), 1024]), length],
                runtimeConstraints=[dict(kind='linear', dimLhs=ref('input', 0, 2),
                                         dimRhs=ref('output', 0, 1), coefficients=[8, residue-8*offset])])
    else:
        return {}
    return {'get_model_schema': json.dumps(schema)}
