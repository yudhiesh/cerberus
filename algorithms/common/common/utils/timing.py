"""Timing utilities for measuring execution time."""

import time
from contextlib import contextmanager
from typing import Generator, Tuple


@contextmanager
def measure_execution_time() -> Generator[None, None, Tuple[float, float]]:
    """
    Context manager to measure execution time.
    
    Returns:
        Tuple of (start_time, elapsed_time_ms)
        
    Example:
        with measure_execution_time() as timer:
            # Do some work
            pass
        start_time, elapsed_ms = timer
    """
    start_time = time.time()
    elapsed_time = 0.0
    
    try:
        yield
    finally:
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return start_time, elapsed_time


class Timer:
    """Simple timer class for measuring execution time."""
    
    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ms 