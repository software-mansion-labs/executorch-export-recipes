"""Shared QNN export helpers."""

# Every Snapdragon part react-native-executorch runs QNN on (its qnnHtpArch in
# cpp/core/utils.cpp), in two files per variant. Each is a Flexible Context
# Binary: one HTP context per SoC, sharing one copy of the weights, and the
# runtime loads the device's. Needs QAIRT >= 2.48 on the device.
#
# Measured on an S26 Ultra (SM8850) against a single-SoC .pte: same latency and
# bit-identical outputs. A SoC left out still loads, but through another SoC's
# context: zipdepth fell back to SM8550's and ran ~20% slower.
#
# v69 gets its own file because it shares almost no weights with v73+: all
# seven in one file is 2.2-6.4x a single-SoC .pte (lraspp a8w8 4.5 -> 29.3 MB),
# while the default group is 1.1-1.45x (6.5 MB) and the v69 file 7.4 MB.
#
# A variant picks its group with `soc_group` in config.json; unset is the
# default group.
SOC_GROUPS = {
    None: ("SM8550", "SM8650", "SM8750", "SM8845", "SM8850"),
    "v69": ("SM8450", "SM8475"),
}


def socs(variant: dict) -> tuple[str, ...]:
    group = variant.get("soc_group")
    if group not in SOC_GROUPS:
        raise ValueError(f"unknown soc_group {group!r} (one of {list(SOC_GROUPS)})")
    return SOC_GROUPS[group]


def compiler_specs(variant: dict, **htp_options):
    """QNN compiler specs for one FCB over the variant's SoCs, HTP options shared."""
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
    )

    group = socs(variant)
    return generate_qnn_executorch_compiler_spec(
        soc_model=[getattr(QcomChipset, soc) for soc in group],
        backend_options=[generate_htp_compiler_spec(**htp_options) for _ in group],
    )


def quantizer_socs(variant: dict) -> list[str]:
    """`make_quantizer(soc_model=...)` for an FCB: QNN validates the quantization
    against the oldest HTP in the list, so it is valid on every target."""
    return list(socs(variant))
