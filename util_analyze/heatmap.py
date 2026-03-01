import matplotlib.pyplot as plt
import os

def plot_layer_head_heatmap(
    values,
    title,
    ax=None,
    vmin=None,
    vmax=None,
    cmap="viridis"
):
    """
    values: np.ndarray [L, H] もしくは [H]（その場合1層扱い）
    """
    if values.ndim == 1:
        values = values[None, :]

    # ax が渡されなければ単体描画（今まで通り）
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(values.shape[1] * 0.6, values.shape[0] * 0.6)
        )
        standalone = True
    else:
        standalone = False

    im = ax.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    # 単体描画のときだけ colorbar & save
    if standalone:
        fig.colorbar(im, ax=ax)
        plt.show()

    return im
