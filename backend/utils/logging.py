# backend/utils/logging.py

import json
import logging
import sys
from datetime import datetime, timezone


# Custom JSON formatter

class JSONFormatter(logging.Formatter):
    """
    Format log records as single-line JSON objects.

    Output example:
    {
      "ts": "2026-04-06T10:23:01.123Z",
      "level": "INFO",
      "logger": "services.parser",
      "message": "Parsing document: LD53657-Carrier-RC.pdf",
      "module": "parser",
      "line": 87
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
            "module":  record.module,
            "line":    record.lineno,
        }

        # Attach exception info if present
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)

        # Attach any extra fields passed via logger.info("...", extra={...})
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            }:
                if not key.startswith("_"):
                    log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False, default=str)


# Human-readable formatter for local development

class DevFormatter(logging.Formatter):
    """
    Coloured, human-readable formatter for local development.
    Format: [HH:MM:SS] LEVEL    logger_name:line — message
    """

    COLOURS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour    = self.COLOURS.get(record.levelname, "")
        ts        = datetime.now().strftime("%H:%M:%S")
        level     = f"{colour}{record.levelname:<8}{self.RESET}"
        location  = f"{record.name}:{record.lineno}"
        message   = record.getMessage()

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return f"[{ts}] {level} {location:<45} — {message}"


# Setup function

# Loggers we want to silence or reduce verbosity on
_NOISY_LOGGERS = {
    "httpx":                  logging.WARNING,
    "httpcore":               logging.WARNING,
    "anthropic":              logging.WARNING,
    "sentence_transformers":  logging.WARNING,
    "transformers":           logging.ERROR,
    "torch":                  logging.ERROR,
    "docling":                logging.WARNING,
    "urllib3":                logging.WARNING,
    "supabase":               logging.WARNING,
    "postgrest":              logging.WARNING,
    "realtime":               logging.WARNING,
}


def setup_logging(debug: bool = False) -> None:
    """Configure root logging for dev or production mode."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove any existing handlers (e.g. added by uvicorn)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)

    if debug:
        handler.setFormatter(DevFormatter())
    else:
        handler.setFormatter(JSONFormatter())

    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for logger_name, level in _NOISY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(level)

    mode = "DEBUG (dev)" if debug else "INFO (production)"
    logging.getLogger(__name__).info(
        f"Logging configured: mode={mode}, format={'dev' if debug else 'json'}"
    )