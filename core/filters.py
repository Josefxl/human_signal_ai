def ema(x, prev, alpha=0.25):
    return alpha * x + (1 - alpha) * prev
