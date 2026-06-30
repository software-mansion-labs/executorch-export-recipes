class ExporterError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class _FixedCodeError(ExporterError):
    _code = "error"

    def __init__(self, message: str):
        super().__init__(self._code, message)


class SpecError(_FixedCodeError):
    _code = "spec_invalid"


class BackendError(_FixedCodeError):
    _code = "backend_unsupported"
