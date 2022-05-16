import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output


class StateActionMesh:
    def __init__(self, states, actions, sleeping_time: float) -> None:
        self.states = states
        self.actions = actions
        self.grid_action, self.grid_states = np.meshgrid(self.actions, self.states)

        self.sleeping_time = sleeping_time

        self.values = np.zeros((len(self.actions), len(self.states)))

    def set_values(self, values: np.ndarray) -> None:
        assert values.shape == (
            len(self.states),
            len(self.actions),
        ), f"given shape values: {values.shape} don't match with environment values: {(len(self.states), len(self.actions))}"

        self.values = values

    def show(self, title: str = "") -> None:
        clear_output(wait=True)
        self.fig, self.ax = plt.subplots()

        colors = self.ax.pcolormesh(self.grid_action, self.grid_states, self.values, shading="nearest")

        self.ax.set_xticks(self.actions)
        self.ax.set_xticklabels(np.around(self.actions, 2))
        self.ax.set_xlim(self.actions[0], self.actions[-1])
        self.ax.set_xlabel("Actions")

        self.ax.set_yticks(self.states)
        self.ax.set_yticklabels(np.around(self.states, 2))
        self.ax.set_ylim(self.states[0], self.states[-1])
        self.ax.set_ylabel("States")

        if title != "":
            self.ax.set_title(title)

        self.fig.colorbar(colors, ax=self.ax)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show()
        time.sleep(self.sleeping_time)
