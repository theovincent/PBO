import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output


class StateActionMesh:
    def __init__(
        self, state_max: float, n_states: int, action_max: float, n_actions: int, sleeping_time: float
    ) -> "StateActionMesh":
        self.n_states = n_states
        self.n_actions = n_actions
        self.sleeping_time = sleeping_time

        self.states = np.linspace(-state_max, state_max, n_states)
        self.actions = np.linspace(-action_max, action_max, n_actions)
        self.grid_action, self.grid_states = np.meshgrid(self.actions, self.states)

        self.values = np.zeros((self.n_actions, self.n_states))

    def set_values(self, values: np.ndarray) -> None:
        assert values.shape == (
            self.n_states,
            self.n_actions,
        ), f"given shape values: {values.shape} don't match with environment values: {(self.n_states, self.n_actions)}"

        self.values = values

    def show(self, title: str = "") -> None:
        clear_output(wait=True)
        self.fig, self.ax = plt.subplots()

        colors = self.ax.pcolormesh(self.grid_action, self.grid_states, self.values, shading="nearest")

        self.ax.set_xticks(self.actions)
        self.ax.set_xticklabels(np.around(self.actions, 2))
        self.ax.set_yticks(self.states)
        self.ax.set_yticklabels(np.around(self.states, 2))
        if title != "":
            self.ax.set_title(title)

        self.fig.colorbar(colors, ax=self.ax)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show()
        time.sleep(self.sleeping_time)
