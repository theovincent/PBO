import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output


class TwoDimesionsMesh:
    def __init__(self, dimension_one, dimension_two, sleeping_time: float) -> None:
        self.dimension_one = dimension_one
        self.dimension_two = dimension_two
        self.grid_dimension_one, self.grid_dimension_two = np.meshgrid(self.dimension_one, self.dimension_two)

        self.sleeping_time = sleeping_time

        self.values = np.zeros((len(self.dimension_one), len(self.dimension_two)))

    def set_values(self, values: np.ndarray, zeros_to_nan: bool = False) -> None:
        assert values.shape == (
            len(self.dimension_one),
            len(self.dimension_two),
        ), f"given shape values: {values.shape} don't match with environment values: {(len(self.dimension_one), len(self.dimension_two))}"

        self.values = values
        if zeros_to_nan:
            self.values = np.where(self.values == 0, np.nan, self.values)

    def show(self, title: str = "", xlabel: str = "States", ylabel: str = "Actions") -> None:
        clear_output(wait=True)
        fig, ax = plt.subplots()

        colors = ax.pcolormesh(self.grid_dimension_one, self.grid_dimension_two, self.values.T, shading="nearest")

        ax.set_xticks(self.dimension_one)
        ax.set_xticklabels(np.around(self.dimension_one, 2))
        ax.set_xlim(self.dimension_one[0], self.dimension_one[-1])
        ax.set_xlabel(xlabel)

        ax.set_yticks(self.dimension_two)
        ax.set_yticklabels(np.around(self.dimension_two, 2))
        ax.set_ylim(self.dimension_two[0], self.dimension_two[-1])
        ax.set_ylabel(ylabel)

        ax.set_aspect("equal", "box")
        if title != "":
            ax.set_title(title)

        fig.colorbar(colors, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        plt.show()
        time.sleep(self.sleeping_time)
