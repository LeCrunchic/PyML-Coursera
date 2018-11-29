from math import floor, ceil, sqrt

import numpy as np
import matplotlib.pyplot as plt
from numpy import ix_

def display_data(x, example_width=None):
    if not example_width:
        example_width = floor(sqrt(x.shape[1]))

    m, n = x.shape
    example_height = int(n / example_width)
    example_width = int(example_width)

    display_rows = floor(sqrt(m))
    display_cols = ceil(m / display_rows)
    pad = 1

    display_array = -np.ones((pad + display_rows * (example_height + pad),
                             pad + display_cols * (example_width + pad)))

    import pdb; pdb.set_trace()
    current_ex = 0
    for j in range(display_rows - 1):
        for i in range(display_cols - 1):
            if current_ex > m:
                break

            max_val = np.max(np.abs(x[current_ex, :]))

            m_idxer = pad + (j-1) * (example_height + pad) + np.arange(0, example_height-1)
            n_idxer = pad + (i-1) * (example_width + pad) + np.arange(0, example_width-1)

            display_array[ix_(m_idxer, n_idxer)] = \
                                  x[current_ex, :].reshape(example_height, example_width) / max_val

            current_ex += 1

        if current_ex > m:
            break

    pdb.set_trace()
    plt.imshow(display_array[-1, 1])
    plt.show()
