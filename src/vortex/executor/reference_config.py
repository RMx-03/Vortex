"""Immutable intended reference settings; validation performs no I/O.

Profiles describe requested execution, not installed or qualified hardware.
The tokenizer repository is deliberately the same fixed repository as the model.
"""

from dataclasses import dataclass

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass(frozen=True)
class ReferenceConfig:
    """Validated scalar settings with token-count limits (never truncation).

    Construction raises ValueError for an unsupported pin, profile or setting.
    Seed zero is part of this reference contract. No environment is probed and
    no revision alias is resolved. Changes require a new configuration instance.
    """

    model_id: str
    revision: str
    tokenizer_revision: str
    profile_id: str
    device: str
    dtype: str
    max_prompt_tokens: int = 512
    max_new_tokens: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        validate_reference_config(self)


def validate_reference_config(config: ReferenceConfig) -> None:
    """Raise ValueError for settings outside the pinned reference contract.

    Both revisions must equal the reviewed full lowercase commit SHA. Limits
    include templated prompt tokens and newly emitted tokens respectively.
    This deterministic CPU-only check never imports inference packages.
    """
    if not isinstance(config, ReferenceConfig):
        raise ValueError("expected ReferenceConfig")
    expected = {"model_id": MODEL_ID, "revision": REVISION, "tokenizer_revision": REVISION}
    for name, value in expected.items():
        actual = getattr(config, name)
        if type(actual) is not str or actual != value:
            raise ValueError(f"unsupported reference {name}")
    profiles = {"windows-cpu": ("cpu", "float32"), "t4-fp16": ("cuda:0", "float16")}
    if type(config.profile_id) is not str or config.profile_id not in profiles:
        raise ValueError("unsupported reference profile_id")
    if (type(config.device) is not str or type(config.dtype) is not str
            or (config.device, config.dtype) != profiles[config.profile_id]):
        raise ValueError("profile requires its explicit device and dtype")
    for name, ceiling in (("max_prompt_tokens", 512), ("max_new_tokens", 64)):
        value = getattr(config, name)
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"invalid reference {name}")
    if type(config.seed) is not int or config.seed != 0:
        raise ValueError("reference seed must be integer zero")
