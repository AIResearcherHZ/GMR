import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_current_dir, "libs")
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

from .libs import taks_driver as _taks_driver

_rl = _taks_driver.rate_limiter

RateLimiter = _rl.RateLimiter
perf_counter = _rl.perf_counter
sleep = _rl.sleep

__all__ = ["RateLimiter", "perf_counter", "sleep"]
