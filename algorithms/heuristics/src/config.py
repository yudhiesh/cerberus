"""Configuration management for heuristics guardrail."""

import os
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from rules_engine import Rule, RuleType


@dataclass
class ServiceConfig:
    """Service configuration."""
    name: str
    version: str
    port: int
    host: str


@dataclass
class RulesEngineConfig:
    """Rules engine configuration."""
    unsafe_threshold: float
    rules: list[Rule]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str


@dataclass
class PerformanceConfig:
    """Performance settings."""
    max_query_length: int
    request_timeout_seconds: int


@dataclass
class FeaturesConfig:
    """Feature flags."""
    log_matches: bool
    return_match_details: bool


@dataclass
class Config:
    """Complete application configuration."""
    service: ServiceConfig
    rules_engine: RulesEngineConfig
    logging: LoggingConfig
    performance: PerformanceConfig
    features: FeaturesConfig


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration using Hydra.
    
    Args:
        config_path: Optional path to config directory. If not provided,
                    uses the default configs directory.
    
    Returns:
        Loaded configuration object
    """
    # Clear any existing Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Determine config directory
    if config_path is None:
        # Default to configs directory relative to this file
        config_path = Path(__file__).parent.parent / "configs"

    # Initialize Hydra with config directory
    with initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None):
        # Compose configuration
        cfg = compose(config_name="config")

        # Convert to Config object
        return _parse_config(cfg)


def _parse_config(cfg: DictConfig) -> Config:
    """Parse OmegaConf configuration into dataclasses."""
    # Parse service config
    service = ServiceConfig(
        name=cfg.service.name,
        version=cfg.service.version,
        port=cfg.service.port,
        host=cfg.service.host
    )

    # Parse rules
    rules = []
    for rule_cfg in cfg.rules_engine.rules:
        rule_type = RuleType(rule_cfg.rule_type)
        rule = Rule(
            name=rule_cfg.name,
            rule_type=rule_type,
            patterns=list(rule_cfg.patterns),
            weight=rule_cfg.weight,
            case_sensitive=rule_cfg.case_sensitive,
            description=rule_cfg.get("description")
        )
        rules.append(rule)

    # Parse rules engine config
    rules_engine = RulesEngineConfig(
        unsafe_threshold=cfg.rules_engine.unsafe_threshold,
        rules=rules
    )

    # Parse other configs
    logging = LoggingConfig(
        level=cfg.logging.level,
        format=cfg.logging.format
    )

    performance = PerformanceConfig(
        max_query_length=cfg.performance.max_query_length,
        request_timeout_seconds=cfg.performance.request_timeout_seconds
    )

    features = FeaturesConfig(
        log_matches=cfg.features.log_matches,
        return_match_details=cfg.features.return_match_details
    )

    return Config(
        service=service,
        rules_engine=rules_engine,
        logging=logging,
        performance=performance,
        features=features
    )


def get_config_from_env() -> Config:
    """
    Load configuration with environment variable overrides.
    
    This allows overriding config values using environment variables
    following Hydra's naming convention.
    """
    # Set any environment-based overrides
    overrides = []

    # Example: HEURISTICS_PORT=8002 would override service.port
    if "HEURISTICS_PORT" in os.environ:
        overrides.append(f"service.port={os.environ['HEURISTICS_PORT']}")

    if "HEURISTICS_UNSAFE_THRESHOLD" in os.environ:
        overrides.append(f"rules_engine.unsafe_threshold={os.environ['HEURISTICS_UNSAFE_THRESHOLD']}")

    # Clear any existing Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize and compose with overrides
    config_path = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
        return _parse_config(cfg)

