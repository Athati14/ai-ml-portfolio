"""Producer/consumer streaming demo. Uses an in-memory queue so it runs with zero infra;
swap queue.Queue for kafka-python's Producer/Consumer for the real thing."""
from __future__ import annotations
import time, queue, threading, random
import numpy as np, joblib

MODEL = joblib.load("fraud_model.joblib")
THRESHOLD = 0.85
bus: "queue.Queue[np.ndarray]" = queue.Queue()


def producer(n=200):
    for _ in range(n):
        bus.put(np.random.randn(20))
        time.sleep(random.uniform(0.005, 0.02))
    bus.put(None)  # poison pill


def consumer():
    flagged = 0
    while True:
        item = bus.get()
        if item is None:
            break
        proba = MODEL.predict_proba(item.reshape(1, -1))[0, 1]
        if proba >= THRESHOLD:
            flagged += 1
            print(f"FLAG p={proba:.3f}")
    print(f"Done. flagged={flagged}")


if __name__ == "__main__":
    t = threading.Thread(target=consumer)
    t.start(); producer(); t.join()
