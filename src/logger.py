import os
import sys
import logging
from datetime import datetime


log_file = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

log_path = os.path.join(os.getcwd(), "logs", log_file)

log_file_path = os.makedirs(log_path, exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging started")
    