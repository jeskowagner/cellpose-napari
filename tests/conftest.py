# torch must be imported before PyQt on Windows to avoid DLL init failure
# https://github.com/pytorch/pytorch/issues/166628
import torch  # noqa: F401
