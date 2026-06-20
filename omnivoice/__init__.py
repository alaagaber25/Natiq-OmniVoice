import warnings
from importlib.metadata import PackageNotFoundError, version

warnings.filterwarnings("ignore", module="torchaudio")
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message="invalid escape sequence",
    module="pydub.utils",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.distributed.algorithms.ddp_comm_hooks",
)


def _patch_webdataset_windows_drive_paths():
    import importlib
    import os

    if os.name != "nt":
        return
    try:
        _g = importlib.import_module("webdataset.gopen")
    except Exception:
        return

    def _local(url, mode="rb", bufsize=8192, **kw):
        return open(url, mode)

    for _c in "abcdefghijklmnopqrstuvwxyz":
        _g.gopen_schemes.setdefault(_c, _local)


_patch_webdataset_windows_drive_paths()

try:
    __version__ = version("omnivoice")
except PackageNotFoundError:
    __version__ = "0.0.0"

from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceConfig,
    OmniVoiceGenerationConfig,
)

__all__ = ["OmniVoice", "OmniVoiceConfig", "OmniVoiceGenerationConfig"]
