import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


class StateActionMesh:
    def __init__(self, action_max: float, n_actions: int, state_max: float, n_states: int) -> "StateActionMesh":
        self.action_max = action_max
        self.n_actions = n_actions
        self.state_max = state_max
        self.n_states = n_states

        self.values = np.zeros((self.n_states, self.n_actions))

    def set_values(self, values: np.ndarray) -> None:
        assert values.shape == (
            self.n_states,
            self.n_actions,
        ), f"given shape values: {values.shape} don't match with environment values: {(self.n_states, self.n_actions)}"

        self.values = values

    def set_title(self, title: str) -> None:
        self.set_title(title)

    def show(self):
        clear_output(wait=True)
        self.fig, self.ax = plt.subplots()

        colors = self.ax.pcolor(self.values)
        self.fig.colorbar(colors, ax=self.ax)
        self.fig.tight_layout()

        self.fig.canvas.draw()
        plt.show()
