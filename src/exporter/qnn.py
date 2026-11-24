"""Shared QNN export helpers."""

# Every Snapdragon part react-native-executorch runs QNN on (its qnnHtpArch in
# cpp/core/utils.cpp). One .pte serves all of them: a Flexible Context Binary
# holds one HTP context per SoC, sharing one copy of the weights, and the
# runtime loads the device's. Needs QAIRT >= 2.48 on the device.
#
# Measured on an S26 Ultra (SM8850) against a single-SoC .pte: same latency and
# bit-identical outputs. A SoC left out of the list still loads, but through
# another SoC's context: zipdepth fell back to SM8550's and ran ~20% slower.
SOCS = ("SM8450", "SM8475", "SM8550", "SM8650", "SM8750", "SM8845", "SM8850")


def compiler_specs(**htp_options):
    """QNN compiler specs for one FCB over `SOCS`, HTP options shared by all."""
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
    )

    return generate_qnn_executorch_compiler_spec(
        soc_model=[getattr(QcomChipset, soc) for soc in SOCS],
        backend_options=[generate_htp_compiler_spec(**htp_options) for _ in SOCS],
    )


def quantizer_socs() -> list[str]:
    """`make_quantizer(soc_model=...)` for an FCB: QNN validates the quantization
    against the oldest HTP in the list, so it is valid on every target."""
    return list(SOCS)
