"""
Simulation logger that writes to both console and a JSON log file
that the dashboard can stream in real-time.

Also installs a global exception hook so ALL uncaught errors across
the entire application get logged to the dashboard.
"""
import json
import os
import sys
import traceback
import threading
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
        # Install global exception hooks
        self._install_hooks()

    def _install_hooks(self):
        """Install global exception hooks so ALL errors get logged."""
        logger = self

        # Hook for uncaught exceptions in main thread
        original_excepthook = sys.excepthook
        def custom_excepthook(exc_type, exc_value, exc_tb):
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            logger.error(f"UNCAUGHT {exc_type.__name__}: {exc_value}")
            logger.error(tb_str)
            original_excepthook(exc_type, exc_value, exc_tb)
        sys.excepthook = custom_excepthook

        # Hook for uncaught exceptions in threads
        original_thread_excepthook = getattr(threading, "excepthook", None)
        def custom_thread_excepthook(args):
            tb_str = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
            logger.error(f"THREAD {args.thread.name} — {args.exc_type.__name__}: {args.exc_value}")
            logger.error(tb_str)
            if original_thread_excepthook:
                original_thread_excepthook(args)
        threading.excepthook = custom_thread_excepthook

    def log(self, level: str, message: str, **kwargs):
        """Write a log entry to file and stdout."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs,
        }
        # Write to file (thread-safe via append mode)
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Don't crash on log write failure
        # Write to stdout with flush
        prefix = {"INFO": "  ", "PHASE": ">>", "METRIC": "**", "ERROR": "!!", "WARN": "⚠️"}
        print(f"{prefix.get(level, '  ')} [{level}] {message}", flush=True)

    def info(self, message: str, **kwargs):
        self.log("INFO", message, **kwargs)

    def phase(self, message: str, **kwargs):
        self.log("PHASE", message, **kwargs)

    def metric(self, message: str, **kwargs):
        self.log("METRIC", message, **kwargs)

    def warn(self, message: str, **kwargs):
        self.log("WARN", message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log("ERROR", message, **kwargs)

    def exception(self, message: str, exc: Exception = None, **kwargs):
        """Log an error with full traceback."""
        self.error(message, **kwargs)
        if exc:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            self.error("".join(tb))
        else:
            self.error(traceback.format_exc())


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
