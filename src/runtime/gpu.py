"""CUDA availability, memory-boundary, and cleanup helpers."""

import torch


class CudaRequiredError(RuntimeError):
    """Raised when a CUDA-only model is requested without an available GPU."""


class GpuMemoryError(RuntimeError):
    """Raised when bounded GPU inference nevertheless exhausts device memory."""

    def __init__(self, operation: str, batch_size: int, sequence_limit: int):
        self.operation = operation
        self.batch_size = batch_size
        self.sequence_limit = sequence_limit
        super().__init__(
            "CUDA out of memory during "
            f"operation={operation} (batch_size={batch_size}, "
            f"sequence_limit={sequence_limit})"
        )


def require_cuda() -> None:
    """Fail before model construction if CUDA inference is unavailable."""
    if not torch.cuda.is_available():
        raise CudaRequiredError(
            "CUDA is required for dense embedding and reranking model execution."
        )


def release_cuda_model(owner: object) -> None:
    """Release an owner's model and best-effort CUDA allocations safely."""
    model = getattr(owner, "model", None)
    owner.model = None
    del model

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (AssertionError, RuntimeError):
        pass
    finally:
        try:
            torch.cuda.empty_cache()
        except (AssertionError, RuntimeError):
            pass
