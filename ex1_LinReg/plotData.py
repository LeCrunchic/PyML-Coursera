import matplotlib.pyplot as plt

def leplot(ax, x , y, param_dict):
    out = ax.plot(x, y, **param_dict)
    return out
