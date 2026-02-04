
import logging, os
from rich.logging import RichHandler

def get_logger(name: str):
    if not logging.getLogger().handlers:
        level = os.getenv("A4S_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(markup=True, rich_tracebacks=False)]
        )
    return logging.getLogger(name)
