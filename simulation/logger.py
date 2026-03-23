"""
Simulation logger that writes to both console and a JSON log file
that the dashboard can stream in real-time.
"""
import json
import os
import sys
from datetime import datetime
from typing import Optional

from config import SIMULATION_DATA_DIR


class SimulationLogger:
    """Logger that writes structured log entries to a file for dashboard consumption."""

    def __init__(self, log_path: Optional[str] = None):
        if log_path is None:
            log_path = os.path.join(SIMULATION_DATA_DIR, "simulation_log.jsonl")
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Clear previous log
        with open(self.log_path, "w") as f:
            pass

    def log(self, level: str, message: str, **kwargs):
        """Write a log entry to file and stdout."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs,
        }
        # Write to file
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        # Write to stdout with flush
        prefix = {"INFO": "  ", "PHASE": ">>", "METRIC": "**", "ERROR": "!!"}
        print(f"{prefix.get(level, '  ')} [{level}] {message}", flush=True)

    def info(self, message: str, **kwargs):
        self.log("INFO", message, **kwargs)

    def phase(self, message: str, **kwargs):
        self.log("PHASE", message, **kwargs)

    def metric(self, message: str, **kwargs):
        self.log("METRIC", message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log("ERROR", message, **kwargs)


# Global logger instance
_logger: Optional[SimulationLogger] = None


def get_logger() -> SimulationLogger:
    global _logger
    if _logger is None:
        _logger = SimulationLogger()
    return _logger


def init_logger(log_path: Optional[str] = None) -> SimulationLogger:
    global _logger
    _logger = SimulationLogger(log_path)
    return _logger
