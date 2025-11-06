"""Configuration management for STS solver"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from pathlib import Path
import json
import os


@dataclass
class SolverConfig:
    """Configuration for a specific solver"""
    name: str
    timeout: int = 300
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """
    Global configuration for STS solver.

    Can be loaded from file or environment variables.
    """

    # Directories
    results_dir: Path = Path("res")
    docs_dir: Path = Path("docs")

    # Default settings
    default_timeout: int = 300
    default_approach: str = "CP"

    # Solver configurations
    solver_configs: Dict[str, SolverConfig] = field(default_factory=dict)

    # Benchmark settings
    benchmark_instances: List[int] = field(default_factory=lambda: list(range(4, 21, 2)))
    benchmark_timeout: int = 300

    # Validation settings
    validate_solutions: bool = True
    official_checker_path: Optional[Path] = None

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from JSON file"""
        with open(path) as f:
            data = json.load(f)
        # parse nested path strings
        if "results_dir" in data:
            data["results_dir"] = Path(data["results_dir"])
        if "docs_dir" in data:
            data["docs_dir"] = Path(data["docs_dir"])
        if "official_checker_path" in data and data["official_checker_path"]:
            data["official_checker_path"] = Path(data["official_checker_path"])
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config = cls()

        if timeout := os.getenv("STS_DEFAULT_TIMEOUT"):
            try:
                config.default_timeout = int(timeout)
            except ValueError:
                pass

        if results_dir := os.getenv("STS_RESULTS_DIR"):
            config.results_dir = Path(results_dir)

        if log_level := os.getenv("STS_LOG_LEVEL"):
            config.log_level = log_level

        return config

    def save(self, path: Path) -> None:
        """Save configuration to file"""
        data = {
            "results_dir": str(self.results_dir),
            "docs_dir": str(self.docs_dir),
            "default_timeout": self.default_timeout,
            "default_approach": self.default_approach,
            "benchmark_instances": self.benchmark_instances,
            "benchmark_timeout": self.benchmark_timeout,
            "validate_solutions": self.validate_solutions,
            "official_checker_path": str(self.official_checker_path) if self.official_checker_path else None,
            "log_level": self.log_level,
            "log_file": str(self.log_file) if self.log_file else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        # Try to load from default locations
        config_paths = [
            Path("sts_config.json"),
            Path.home() / ".sts_config.json",
        ]

        for path in config_paths:
            if path.exists():
                _config = Config.from_file(path)
                break
        else:
            # Load from environment or use defaults
            _config = Config.from_env()

    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance"""
    global _config
    _config = config
