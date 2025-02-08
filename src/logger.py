import os
import sys
import logging
from datetime import datetime


log_file = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

log_path = os.path.join(os.getcwd(), "logs")

os.makedirs(log_path, exist_ok=True)

log_file_path = os.path.join(log_path, log_file)

logging.basicConfig(
    filename=log_file_path,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging started")
    