"""
Suppress noisy HuggingFace import warnings (must be imported before transformers).

When sentence-transformers loads transformers, recent versions scan every
image-processing submodule and print hundreds of `__path__` deprecation lines.
"""

import logging
import os
import warnings

_PATH_MSG = "Accessing `__path__`"


class _SuppressPathMessages(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _PATH_MSG not in record.getMessage()


_filter = _SuppressPathMessages()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
logging.captureWarnings(True)
logging.getLogger("py.warnings").addFilter(_filter)
logging.root.addFilter(_filter)

for _logger in (
    "transformers",
    "sentence_transformers",
    "torch",
    "huggingface_hub",
):
    logging.getLogger(_logger).setLevel(logging.ERROR)


class _FilteredStderr:
    """Drop HuggingFace lazy-module lines that bypass the warnings module."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text: str) -> int:
        if _PATH_MSG in text:
            return len(text)
        return self._stream.write(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_stderr_filter():
    import sys

    if not isinstance(sys.stderr, _FilteredStderr):
        sys.stderr = _FilteredStderr(sys.stderr)


_install_stderr_filter()
