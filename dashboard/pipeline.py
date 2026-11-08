"""Raw firmware message -> one feature row, exactly as pm/dataset.py builds training rows.

pm/dataset.session_rows() does, for every message of a recorded session:
    feats = features.extract(msg)            # ~420 per-message features (vibration, audio, ...)
    history.append(feats)                    # last TEMPORAL_WINDOW (15) messages = 30 s
    feats = {**feats, **features.temporal(history)}   # tm_* trend features
LivePipeline does the same thing on the live stream, one history per device, so a model sees the
same columns live as it saw in training.
"""
import math
from collections import deque

from common import features


class LivePipeline:
    def __init__(self, gap_reset_s=30.0):
        self.history = {}           # device_id -> deque of feature dicts
        self.last_uptime = {}       # device_id -> uptime_ms of the previous message
        self.gap_reset_ms = gap_reset_s * 1000

    def reset(self, device=None):
        if device is None:
            self.history.clear()
            self.last_uptime.clear()
        else:
            self.history.pop(device, None)
            self.last_uptime.pop(device, None)

    def __call__(self, msg):
        """Return (feature dict, history list) or (None, None) if the message has no vibration burst."""
        feats = features.extract(msg)
        if feats is None:
            return None, None
        dev = msg.get("device_id", "unknown")
        up = msg.get("uptime_ms")
        prev = self.last_uptime.get(dev)
        # ESP32 restarted (uptime went backwards) or a long gap: the 30 s trend window is stale.
        if isinstance(up, (int, float)) and prev is not None and (up < prev or up - prev > self.gap_reset_ms):
            self.history.pop(dev, None)
        if isinstance(up, (int, float)):
            self.last_uptime[dev] = up
        hist = self.history.setdefault(dev, deque(maxlen=features.TEMPORAL_WINDOW))
        hist.append(feats)
        row = {**feats, **features.temporal(list(hist))}
        return row, list(hist)


def clean(v):
    """float or None (NaN/inf/non-numbers -> None), for JSON."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f
