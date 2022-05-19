import numpy as np
import matplotlib.pyplot as plt


def shared_axis_plot(
    ax1_data: np.array, ax2_data: np.array, xlabel: str = "", ax1_title: str = "", ax2_title: str = ""
) -> None:
    assert len(ax1_data) == len(
        ax2_data
    ), f"{ax1_title} has length of {len(ax1_data)} and {ax2_title} has length of {len(ax2_data)}"
    fig, ax1 = plt.subplots()

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ax1_title, color="blue")
    ax1.plot(np.arange(len(ax1_data)), ax1_data, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel(ax2_title, color="red")
    ax2.plot(np.arange(len(ax2_data)), ax2_data, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.tight_layout()
    plt.show()
