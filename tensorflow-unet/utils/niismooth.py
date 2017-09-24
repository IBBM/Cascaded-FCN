import numpy as np

import scipy
import scipy.signal

class Smoothutil:
    """
    Uses 2d convolution to smooth labels. Sometimes it is better to also take adjacent pixels into account.
    """
    def __init__(self, kernel=None, n_kernel_padd=0, pad_value=1):
        self.kernel = kernel

        if kernel is None:
            self.kernel = np.array([
                        [3, 3, 3],
                        [3, 7, 3],
                        [3, 3, 3],
                    ], dtype=np.float32)

        self.kernel = np.pad(self.kernel, ((n_kernel_padd,n_kernel_padd), (n_kernel_padd,n_kernel_padd)), mode='constant', constant_values=pad_value)

    def convolve2d(self, predictions):
        return scipy.signal.convolve2d(predictions, self.kernel, mode="same") / self.kernel.sum()
