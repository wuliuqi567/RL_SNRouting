import csv
import os
from typing import Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd


RECEIVED_BLOCK_COLUMNS = [
    "Creation Time",
    "Latency",
    "Arrival Time",
    "Source",
    "Destination",
    "Block ID",
    "QueueTime",
    "TxTime",
    "PropTime",
]


class CountingList:
    """
    List-like counter that optionally retains appended objects.

    This keeps the old append/clear/len calling style intact while allowing us
    to avoid holding every block object in memory for statistics that only need
    a count.
    """

    def __init__(self, retain_items: bool = False):
        self.retain_items = retain_items
        self._items: List[object] = []
        self._count = 0

    def append(self, item):
        self._count += 1
        if self.retain_items:
            self._items.append(item)

    def clear(self):
        self._count = 0
        self._items.clear()

    def __len__(self):
        return self._count

    def __iter__(self) -> Iterator[object]:
        return iter(self._items)

    def items(self) -> List[object]:
        return list(self._items)


class ReceivedBlockStore:
    """
    Store finished-block statistics.

    In legacy mode, finished block objects are retained in memory.
    In streaming mode, only a small row buffer is kept in memory and flushed to
    CSV once it reaches a configured threshold.
    """

    def __init__(self):
        self._items: List[object] = []
        self._buffer: List[dict] = []
        self._count = 0
        self._stream_to_disk = False
        self._flush_threshold = 5000
        self._csv_path: Optional[str] = None

    @property
    def csv_path(self) -> Optional[str]:
        return self._csv_path

    @property
    def stream_to_disk(self) -> bool:
        return self._stream_to_disk

    def configure(self, output_path: str, stream_to_disk: bool = False, flush_threshold: int = 5000):
        self.clear()
        self._stream_to_disk = bool(stream_to_disk)
        self._flush_threshold = max(1, int(flush_threshold))
        self._csv_path = None

        if not self._stream_to_disk:
            return

        csv_dir = os.path.join(output_path, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        self._csv_path = os.path.join(csv_dir, "received_block_stats.csv")
        with open(self._csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RECEIVED_BLOCK_COLUMNS)
            writer.writeheader()

    def append(self, block):
        self._count += 1
        if not self._stream_to_disk:
            self._items.append(block)
            return

        self._buffer.append(self._serialize_block(block))
        if len(self._buffer) >= self._flush_threshold:
            self.flush()

    def flush(self):
        if not self._stream_to_disk or not self._buffer or self._csv_path is None:
            return

        with open(self._csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RECEIVED_BLOCK_COLUMNS)
            writer.writerows(self._buffer)
        self._buffer.clear()

    def finalize(self):
        self.flush()

    def clear(self):
        self._items.clear()
        self._buffer.clear()
        self._count = 0

    def __len__(self):
        return self._count

    def __iter__(self) -> Iterator[object]:
        return iter(self._items)

    def to_dataframe(self) -> pd.DataFrame:
        if self._stream_to_disk and self._csv_path and os.path.exists(self._csv_path):
            self.flush()
            return pd.read_csv(self._csv_path)

        if not self._items:
            return pd.DataFrame(columns=RECEIVED_BLOCK_COLUMNS)

        rows = [self._serialize_block(block) for block in self._items]
        return pd.DataFrame(rows, columns=RECEIVED_BLOCK_COLUMNS)

    def _serialize_block(self, block) -> dict:
        total_time = float(block.getTotalTransmissionTime())
        queue_time = float(block.getQueueTime()[0])
        source_name = getattr(block.source, "name", str(block.source))
        destination_name = getattr(block.destination, "name", str(block.destination))
        return {
            "Creation Time": float(block.creationTime),
            "Latency": total_time,
            "Arrival Time": float(block.creationTime + total_time),
            "Source": source_name,
            "Destination": destination_name,
            "Block ID": block.ID,
            "QueueTime": queue_time,
            "TxTime": float(block.txLatency),
            "PropTime": float(block.propLatency),
        }


receivedDataBlocks = ReceivedBlockStore()
createdBlocks = CountingList(retain_items=False)
seed = np.random.seed(1)
upGSLRates = []
downGSLRates = []
interRates = []
intraRate = []
dropBlocks = CountingList(retain_items=False)


def configure_runtime_stats(output_path: str, stream_to_disk: bool = False, flush_threshold: int = 5000):
    receivedDataBlocks.configure(
        output_path=output_path,
        stream_to_disk=stream_to_disk,
        flush_threshold=flush_threshold,
    )


def finalize_runtime_stats():
    receivedDataBlocks.finalize()

