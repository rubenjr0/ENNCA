from matplotlib import pyplot as plt
import numpy as np


def visualize(name, batch, model=None, n=4):
    x, y = batch
    n = min(n, len(x))
    x = x[0:n]
    y = y[0:n]
    if model:
        fig, ax = plt.subplots(nrows=3, ncols=n)
    else:
        fig, ax = plt.subplots(nrows=2, ncols=n)
    fig.suptitle(name)
    for i, axi in enumerate(ax.flat):
        axi.axis(False)
        image = x[i % n]
        target = y[i % n].squeeze()
        if i < n:
            image = image.permute(1, 2, 0).cpu().numpy()
            axi.imshow((image * 255).astype(np.uint8), cmap='gray')
            if i % n == 0:
                axi.set_title('original image')
        elif i < 2*n:
            axi.imshow(target.cpu(), cmap='gray')
            if i % n == 0:
                axi.set_title('real depth')
        else:
            image = image.unsqueeze(0)
            estimated = model(image).detach().squeeze().squeeze()
            axi.imshow(estimated.cpu(), cmap='gray')
            if i % n == 0:
                axi.set_title('estimated depth')
    fig.tight_layout()
    fig.savefig(name)
    return fig
