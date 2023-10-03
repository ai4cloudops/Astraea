import logging
import os
import sys

def setup_logging():
    os.makedirs("results", exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO
    )
    return logger
