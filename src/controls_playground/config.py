"""YAML config loader with attribute-style access."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


class Config(SimpleNamespace):
    """SimpleNamespace that recursively converts nested dicts."""

    @classmethod
    def from_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return cls(**{k: cls.from_dict(v) for k, v in data.items()})
        if isinstance(data, list):
            return [cls.from_dict(v) for v in data]
        return data

    def to_dict(self) -> dict:
        out: dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                out[k] = v
        return out


def load(path: str | Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config.from_dict(data)
