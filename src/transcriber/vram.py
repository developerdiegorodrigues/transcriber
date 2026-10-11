"""VRAM budgeting for the faster-whisper backend.

The GPU on this machine is routinely shared with other AI workloads, so neither the
total memory nor a fixed profile value describes what is actually available. These
helpers turn the *free* memory into a starting batch size, and explain an allocation
failure in terms the user can act on.

The constants are coarse heuristics, deliberately conservative: the batch retreat in
the backend remains the safety net, and benchmark data refines them (TODO 1.3).
"""

from __future__ import annotations

from dataclasses import dataclass


#: Approximate FP16 weight footprint per model, in MiB.
_MODEL_WEIGHTS_MIB: dict[str, int] = {
    "tiny": 75,
    "tiny.en": 75,
    "base": 145,
    "base.en": 145,
    "small": 480,
    "small.en": 480,
    "medium": 1530,
    "medium.en": 1530,
    "large": 3090,
    "large-v1": 3090,
    "large-v2": 3090,
    "large-v3": 3090,
    "turbo": 1610,
    "large-v3-turbo": 1610,
    "distil-small.en": 330,
    "distil-medium.en": 780,
    "distil-large-v2": 1510,
    "distil-large-v3": 1510,
}

#: Approximate activation cost of one batch slot, in MiB. Driven by the encoder, so
#: `turbo` costs as much per slot as `large-v3` despite the smaller decoder.
_SLOT_COST_MIB: dict[str, int] = {
    "tiny": 25,
    "tiny.en": 25,
    "base": 40,
    "base.en": 40,
    "small": 80,
    "small.en": 80,
    "medium": 160,
    "medium.en": 160,
    "large": 260,
    "large-v1": 260,
    "large-v2": 260,
    "large-v3": 260,
    "turbo": 260,
    "large-v3-turbo": 260,
    "distil-small.en": 80,
    "distil-medium.en": 160,
    "distil-large-v2": 260,
    "distil-large-v3": 260,
}

#: CUDA context, cuBLAS workspaces and cuDNN scratch, paid once per process.
CONTEXT_OVERHEAD_MIB = 600

#: Never plan to use the last slice of free memory: another process may grow.
SAFETY_MARGIN_MIB = 320

#: Measured ceiling, not a memory ceiling. On the RTX 5060 Ti with the GPU idle,
#: throughput peaks at batch 4-8 and then collapses: `large-v3` took 26.8 s at batch 4
#: and 54.3 s at batch 32 over the same 1120 s of audio, and `turbo` at batch 32 was
#: slower than batch 1. Free VRAM therefore buys headroom for a bigger *model*, not a
#: bigger batch. See reference/Progress_Control.md, D14.
MAXIMUM_AUTOMATIC_BATCH = 8


def _weight_scale(compute_type: str) -> float:
    if compute_type.startswith("int8"):
        return 0.5
    if compute_type == "float32":
        return 2.0
    return 1.0


def model_weights_mib(model: str, compute_type: str) -> int | None:
    """Estimate the weight footprint, or ``None`` for an unknown model."""

    base = _MODEL_WEIGHTS_MIB.get(model)
    if base is None:
        return None
    return round(base * _weight_scale(compute_type))


def slot_cost_mib(model: str, compute_type: str) -> int | None:
    base = _SLOT_COST_MIB.get(model)
    if base is None:
        return None
    return round(base * _weight_scale(compute_type))


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """A starting batch size and the reason behind it."""

    batch_size: int
    reason: str
    fits: bool = True


def plan_batch_size(
    requested: int,
    *,
    free_mib: int | None,
    model: str,
    compute_type: str,
    explicit: bool,
    maximum: int = MAXIMUM_AUTOMATIC_BATCH,
) -> BatchPlan:
    """Fit the starting batch size to the memory that is actually free.

    Shrinks when another workload holds the card and never grows past the profile
    value, because measurement showed a larger batch costs throughput rather than
    gaining it. An explicit ``--batch-size`` is never overridden: the user's option
    always wins over the profile and over this estimate.
    """

    if explicit:
        return BatchPlan(requested, "batch definido explicitamente")
    if free_mib is None:
        return BatchPlan(requested, "VRAM livre desconhecida; mantendo o batch do perfil")

    weights = model_weights_mib(model, compute_type)
    slot = slot_cost_mib(model, compute_type)
    if weights is None or slot is None:
        return BatchPlan(
            requested,
            f"modelo '{model}' não catalogado; mantendo o batch do perfil",
        )

    usable = free_mib - CONTEXT_OVERHEAD_MIB - SAFETY_MARGIN_MIB
    if usable < weights:
        return BatchPlan(
            1,
            f"apenas {free_mib} MiB livres para ~{weights} MiB de pesos mais "
            f"{CONTEXT_OVERHEAD_MIB} MiB de contexto CUDA",
            fits=False,
        )

    # Only ever shrink: growing past the profile value was measured as a regression,
    # so abundant VRAM is not a reason to raise the batch.
    affordable = max(1, min(requested, maximum, (usable - weights) // slot))
    if affordable < requested:
        return BatchPlan(
            affordable,
            f"{free_mib} MiB livres exigem reduzir de {requested} para {affordable}",
        )
    return BatchPlan(requested, f"{free_mib} MiB livres acomodam o batch {requested}")


def describe_allocation_failure(
    *, free_mib: int | None, model: str, compute_type: str, device_total_mib: int | None = None
) -> str:
    """Explain an out-of-memory failure and offer concrete ways out."""

    weights = model_weights_mib(model, compute_type)
    parts = [f"VRAM insuficiente para o modelo '{model}' em {compute_type}"]
    if free_mib is not None:
        occupied = (
            f" de {device_total_mib} MiB" if device_total_mib is not None else ""
        )
        parts.append(f"havia {free_mib} MiB livres{occupied}")
    if weights is not None:
        parts.append(
            f"os pesos exigem ~{weights} MiB mais ~{CONTEXT_OVERHEAD_MIB} MiB de contexto CUDA"
        )
    alternatives = [
        "libere a GPU (outro processo pode estar ocupando a placa)",
        "use um modelo menor com --model",
        "reduza a precisão com --compute-type int8_float16",
        "ou transcreva na CPU com --device cpu",
    ]
    return f"{'; '.join(parts)}. Alternativas: {', '.join(alternatives)}."
