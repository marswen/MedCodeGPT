import json
import uuid
import threading
from datetime import datetime
from loguru import logger

logger.add("logs/debug.log", rotation="00:00")


class DetailLogger:
    def __init__(self):
        self.mutex = threading.Lock()

    def add_record(self, *data):
        with self.mutex:
            record = {'uuid': uuid.uuid4().hex,
                      'timestamp': str(datetime.now()),
                      'user_input': data[0],
                      'output': data[1]}
            with open('logs/details.jsonl', 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write('\n')
