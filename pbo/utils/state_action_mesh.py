import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output


class StateActionMesh:
    def __init__(self, states, actions, sleeping_time: float) -> None:
        self.states = states
        self.actions = actions
        self.grid_states, self.grid_action = np.meshgrid(self.states, self.actions)

        self.sleeping_time = sleeping_time

        self.values = np.zeros((len(self.states), len(self.actions)))

    def set_values(self, values: np.ndarray, zeros_to_nan: bool = False) -> None:
        assert values.shape == (
            len(self.states),
            len(self.actions),
        ), f"given shape values: {values.shape} don't match with environment values: {(len(self.states), len(self.actions))}"

        self.values = values
        if zeros_to_nan:
            self.values = np.where(self.values == 0, np.nan, self.values)

    def show(self, title: str = "") -> None:
        clear_output(wait=True)
        fig, ax = plt.subplots()

        colors = ax.pcolormesh(self.grid_states, self.grid_action, self.values.T, shading="nearest")

        ax.set_xticks(self.states)
        ax.set_xticklabels(np.around(self.states, 2))
        ax.set_xlim(self.states[0], self.states[-1])
        ax.set_xlabel("States")

        ax.set_yticks(self.actions)
        ax.set_yticklabels(np.around(self.actions, 2))
        ax.set_ylim(self.actions[0], self.actions[-1])
        ax.set_ylabel("Actions")

        ax.set_aspect("equal", "box")
        if title != "":
            ax.set_title(title)

        fig.colorbar(colors, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        plt.show()
        time.sleep(self.sleeping_time)
