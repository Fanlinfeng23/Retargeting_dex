# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import zmq
import numpy as np
import threading
import time

class ManusMocap:
    '''
    Applies to any ZMQ-broadcasted mocap data with fixed shape (21,3) and dtype float32.
    Runs a background thread to continuously receive and update latest data.
    '''
    def __init__(self, port=8765):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://localhost:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        socket.setsockopt(zmq.RCVHWM, 1)
        self.socket = socket
        self.context = context

        self._latest_data = None
        self._latest_timestamp = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            # Use NOBLOCK + Python sleep instead of zmq.poll(blocking).
            # zmq.poll() is a C-level blocking syscall that conflicts with
            # SAPIEN's Vulkan renderer on the main thread and causes a segfault.
            # Python's time.sleep() uses nanosleep() which does not conflict.
            try:
                msg = self.socket.recv(flags=zmq.NOBLOCK)
                arr = np.frombuffer(msg, dtype=np.float32).reshape(21, 3)
                if not np.isfinite(arr).all():
                    continue
                with self._lock:
                    self._latest_data = arr
                    self._latest_timestamp = time.monotonic()
            except zmq.Again:
                time.sleep(0.005)  # 5ms Python sleep; drain queue fast when data flows
            except ValueError:
                # Ignore malformed frames instead of killing the receive thread.
                continue
            except zmq.ZMQError:
                if self._running:
                    raise
                break

    def get(self, stale_after=0.5):
        with self._lock:
            if self._latest_data is not None and self._latest_timestamp is not None:
                age = time.monotonic() - self._latest_timestamp
                if age <= stale_after:
                    return {"result": self._latest_data.copy(), "status": "recording"}
                return {"result": None, "status": "stale"}
            else:
                return {"result": None, "status": "no data"}

    def close(self):
        self._running = False
        self._thread.join()
        self.socket.close(0)
        self.context.term()

    def __del__(self):
        if getattr(self, "_running", False):
            self.close()
