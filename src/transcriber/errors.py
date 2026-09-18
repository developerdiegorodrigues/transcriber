"""Domain exceptions with messages suitable for the command line."""


class TranscriberError(RuntimeError):
    """Base exception for expected application failures."""


class HardwareError(TranscriberError):
    """Raised when the requested compute device is unavailable."""


class DependencyError(TranscriberError):
    """Raised when an external runtime dependency is unavailable."""


class TranscriptionError(TranscriberError):
    """Raised when the transcription backend fails."""
