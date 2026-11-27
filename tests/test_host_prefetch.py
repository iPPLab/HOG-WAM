import numpy as np
import pytest

from fastwam_tpu.data.pool import stable_seed
from fastwam_tpu.training.host_prefetch import HostBatchPrefetch


@pytest.mark.parametrize('accum', [1, 4, 8])
@pytest.mark.parametrize('start', [1, 251])
def test_same_batch_stream_including_resume(accum, start):
    def rng_for(step, micro):
        return np.random.default_rng(stable_seed(42, 'batch', 0, step, micro))

    def sample(rng, size):
        return {'rows': rng.integers(0, 337919, size=size),
                'pixels': rng.standard_normal((size, 2, 3)).astype(np.float32)}

    prefetch = HostBatchPrefetch(sample, rng_for, 8, start, accum)
    try:
        for step in range(start, start + 3):
            for micro in range(accum):
                expected = sample(rng_for(step, micro), 8)
                actual = prefetch.get(step, micro)
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])
    finally:
        prefetch.close()


def test_rejects_out_of_order_consumption():
    prefetch = HostBatchPrefetch(lambda rng, n: rng.random(n),
                                lambda s, m: np.random.default_rng(s + m), 2, 1, 4)
    try:
        with pytest.raises(ValueError, match='Out-of-order'):
            prefetch.get(1, 1)
    finally:
        prefetch.close()
