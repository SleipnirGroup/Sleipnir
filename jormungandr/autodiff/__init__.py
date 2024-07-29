from .._jormungandr.autodiff import *

import scipy.sparse


def spy(mat: scipy.sparse.csc_matrix):
    """
    Plot the sparsity pattern of a sparse matrix.

    Green points represent positive values and red points represent negative
    values.

    Parameter ``mat``:
        The sparse matrix.
    """
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    xs = []
    ys = []
    vs = []

    cmat = mat.tocoo()
    for row, col, value in zip(cmat.row, cmat.col, cmat.data):
        xs.append(col)
        ys.append(row)
        if value > 0.0:
            vs.append(1.0)
        elif value < 0.0:
            vs.append(-1.0)
        else:
            vs.append(0.0)

    fig = plt.figure()
    ax = fig.add_subplot()

    # Display scatter plot
    cmap = ListedColormap(["red", "green"])
    sc = ax.scatter(xs, ys, s=1, c=vs, marker=".", cmap=cmap, vmin=-1, vmax=1)

    # Display colorbar
    colorbar = fig.colorbar(sc)
    ticklabels = ["Negative", "Positive"]
    colorbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticklabels))
    colorbar.ax.set_yticks([-1, 1])
    colorbar.ax.set_yticklabels(ticklabels)

    ax.set_title("Sparsity")
    ax.set_xlabel("Cols")
    ax.set_ylabel("Rows")

    ax.set_xlim([0, mat.shape[1]])
    ax.set_ylim([0, mat.shape[0]])
    ax.invert_yaxis()
    ax.set_aspect(1.0)

    plt.show()
