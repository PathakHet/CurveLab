"""Run configuration. Every tunable parameter lives here or in a YAML override."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    market_dir: str = "data/sample"
    journals: tuple[str, ...] = ("data/sample/Sample Trade Journal.xlsx",)
    min_bars: int = 700
    max_ffill_bars: int = 1


@dataclass
class UniverseConfig:
    spacings: tuple[int, ...] = (1, 2, 3)
    kinds: tuple[str, ...] = ("spread", "fly", "dfly")
    min_structure_bars: int = 500


@dataclass
class ResearchConfig:
    horizon: int = 6
    zscore_window: int = 48
    vol_window: int = 24
    train_bars: int = 400
    test_bars: int = 120
    step_bars: int = 120
    embargo_bars: int = 6
    top_k: int = 5
    max_active: int = 10
    hysteresis: float = 0.25
    model: str = "hist_gradient_boosting"
    periods_per_year: float = 6000.0
    delays: tuple[int, ...] = (0, 1, 2, 3)


@dataclass
class CostConfig:
    cost_per_leg: float = 0.01
    slippage_per_leg: float = 0.005
    contract_multiplier: float = 1000.0


@dataclass
class DeskConfig:
    fdr_q: float = 0.10
    min_days: int = 8
    bootstrap_draws: int = 2000
    anonymize: bool = True
    anonymize_salt: str = "curvelab"
    mapping_path: str = "~/.curvelab/trader_mapping.json"


@dataclass
class OutputConfig:
    results_dir: str = "reports/results"
    figures_dir: str = "reports/figures"
    report_path: str = "reports/REPORT.md"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    desk: DeskConfig = field(default_factory=DeskConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load a config, overlaying a YAML file onto the defaults if given."""
        config = cls()
        if path is None:
            return config
        import yaml

        overrides = yaml.safe_load(Path(path).read_text()) or {}
        return _merge(config, overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(config: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(config, key, tuple(value))
        else:
            setattr(config, key, value)
    return config
