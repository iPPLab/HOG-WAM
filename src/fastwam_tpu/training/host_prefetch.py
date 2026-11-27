"""One-batch deterministic CPU look-ahead, including gradient accumulation."""
from concurrent.futures import ThreadPoolExecutor


class HostBatchPrefetch:
    """Read exactly the same (step, micro) stream as the synchronous loader.

    The worker only produces host numpy arrays. JAX and all GPU operations remain
    on the training thread. A single future bounds memory and preserves ordering.
    """

    def __init__(self, sample, rng_for, batch_size, start_step, accum):
        if accum < 1:
            raise ValueError('accum must be positive')
        self.sample = sample
        self.rng_for = rng_for
        self.batch_size = batch_size
        self.accum = accum
        self.expected = (start_step, 0)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(sample, rng_for(start_step, 0), batch_size)

    def get(self, step, micro):
        if (step, micro) != self.expected:
            raise ValueError(f'Out-of-order batch: {(step, micro)} != {self.expected}')
        result = self.future.result()
        next_step, next_micro = ((step, micro + 1) if micro + 1 < self.accum
                                 else (step + 1, 0))
        self.expected = (next_step, next_micro)
        self.future = self.executor.submit(
            self.sample, self.rng_for(next_step, next_micro), self.batch_size)
        return result

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
