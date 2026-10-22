"""In-memory dataset sessions (no uploaded data is written to disk).

Every dataset belongs to the browser session that uploaded it (`owner`, a random
token kept in an HttpOnly cookie). Lookups with any other owner behave exactly
like a missing dataset, so a dataset id alone — e.g. from a copied link — never
gives anyone else access to the data.
"""
import os
import threading
import time
import uuid
from collections import OrderedDict

MAX_DATASETS = int(os.environ.get("SDAP_MAX_DATASETS", "40"))          # all visitors together (memory bound)
MAX_PER_OWNER = int(os.environ.get("SDAP_MAX_PER_SESSION", "3"))       # one visitor's open datasets
TTL_SECONDS = int(os.environ.get("SDAP_DATASET_TTL", str(3 * 3600)))


class DatasetStore:
    def __init__(self, max_items=MAX_DATASETS, ttl=TTL_SECONDS, max_per_owner=MAX_PER_OWNER):
        self._items = OrderedDict()       # id -> (owner, dataset), least recently used first
        self._lock = threading.Lock()
        self.max_items = max_items
        self.max_per_owner = max_per_owner
        self.ttl = ttl

    @staticmethod
    def new_id():
        return uuid.uuid4().hex

    def put(self, dataset, owner=None):
        with self._lock:
            self._evict()
            self._items[dataset.id] = (owner, dataset)
            self._items.move_to_end(dataset.id)
            # A visitor's own oldest datasets go first, so one visitor cannot push out the others
            mine = [k for k, (o, _) in self._items.items() if o == owner]
            for k in mine[:max(0, len(mine) - self.max_per_owner)]:
                del self._items[k]
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def get(self, dataset_id, owner=None):
        with self._lock:
            self._evict()
            item = self._items.get(dataset_id)
            if item is None or item[0] != owner:
                return None
            item[1].last_access = time.time()
            self._items.move_to_end(dataset_id)
            return item[1]

    def delete(self, dataset_id, owner=None):
        with self._lock:
            item = self._items.get(dataset_id)
            if item is None or item[0] != owner:
                return False
            del self._items[dataset_id]
            return True

    def _evict(self):
        now = time.time()
        for k in [k for k, (_, v) in self._items.items() if now - v.last_access > self.ttl]:
            del self._items[k]
