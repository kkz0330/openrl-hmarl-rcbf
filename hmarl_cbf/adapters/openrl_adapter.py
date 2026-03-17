from __future__ import annotations

from typing import Any, Dict


class OpenRLAdapter:
    """Reserved adapter surface for future OpenRL integration."""

    def __init__(self, env: Any, high_policy: Any, low_policy: Any) -> None:
        self.env = env
        self.high_policy = high_policy
        self.low_policy = low_policy

    def to_openrl_env(self) -> Any:
        raise NotImplementedError("OpenRL env adapter is reserved for a later integration step")

    def to_openrl_policy_io(self) -> Dict[str, Any]:
        raise NotImplementedError("OpenRL policy adapter is reserved for a later integration step")
