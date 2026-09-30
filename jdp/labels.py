"""Pixel click <-> training target, in both conventions.

2024:      x = (u - 50) / 50,  y = (v - 50) / 50   (rf_train_model.ipynb get_x/get_y; a
           holdover from an older NVIDIA jetbot notebook, applied to 224-px clicks)
symmetric: x = (u - 112) / 112, y = (v - 112) / 112 (what NVIDIA's notebook uses today)

Flip augmentation in 2024 ran on every sample at p=0.5 regardless of the random_hflips
flag, and negated x. Under the symmetric convention that is right to within half a pixel; under the 2024
convention the mirrored click is 2*(111.5-50)/50 - x = 2.46 - x, so -x is wrong by 2.46.
"""
OUT = 224


def encode(u, v, mode):
    if mode == "2024":
        return (u - 50.0) / 50.0, (v - 50.0) / 50.0
    if mode == "2024_flipmean":
        # Not a training convention: the best a perfectly trained 2024 net can output. Under the
        # always-on 50% flip with x = -x, the MSE-optimal x is the mean of (u-50)/50 and
        # -((223-u)-50)/50, i.e. (u-111.5)/50. y is never flipped, so it keeps (v-50)/50.
        return (u - 111.5) / 50.0, (v - 50.0) / 50.0
    if mode == "symmetric":
        c = OUT / 2
        return (u - c) / c, (v - c) / c
    raise ValueError(mode)


def decode(x, y, mode):
    if mode == "2024":
        return x * 50.0 + 50.0, y * 50.0 + 50.0
    c = OUT / 2
    return x * c + c, y * c + c


def flip_x(x, mode, faithful):
    """x after a horizontal image flip. faithful=True reproduces the 2024 `x = -x`."""
    if faithful or mode == "symmetric":
        return -x
    u, _ = decode(x, 0.0, mode)
    return encode(OUT - 1 - u, 0.0, mode)[0]
