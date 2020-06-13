from typing import Dict, Tuple, Optional

from highway_env.envs.common.abstract import Action, Observation
from highway_env.road.objects import Landmark
import numpy as np

from tl_env.envs.single_goal import SingleGoalIDMEnv
from tl_env.logic.automaton import Automaton


class DoubleGoalEnv(SingleGoalIDMEnv):
    """A continuous control environment with two goals and some adversary vehicles.

    The vehicle must reach the goals while avoiding other vehicles.
    """
    def __init__(self, config: Optional[Dict] = None) -> None:
        self.automaton = None
        super().__init__(config)

    def default_config(self) -> Dict:
        config = super().default_config()
        config.update(
            {
                "duration": 50,
                "goal2_position": [320, 0]
            }
        )
        return config

    def reset(self) -> np.ndarray:
        self._create_automaton()
        return super().reset()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        obs, reward, terminal, info = super().step(action)
        # step the automaton with newly updated info about reaching goals
        self.automaton.step(self._goal_achievement())
        # update info and terminal with new automaton state
        info.update(dict(q=self.automaton.cur_state))
        terminal = self._is_terminal()
        return obs, reward, terminal, info

    def _create_road(self) -> None:
        super()._create_road()
        self.goal2 = Landmark(self.road, self.config["goal2_position"], heading=0)
        self.road.objects.append(self.goal2)

    def _create_automaton(self) -> None:
        self.automaton = Automaton()
        self.automaton.add_state_from(
            [
                ('q0', {'type': 'init'}),
                'q1',
                ('q2', {'type': 'final'})
            ]
        )
        self.automaton.add_transition_from(
            [
                ('q0', 'q1', 'g1'),
                ('q1', 'q2', 'g2')
            ]
        )

    def _goal_achievement(self) -> Dict:
        # noinspection PyProtectedMember
        return {
            'g1': self.vehicle._is_colliding(self.goal),
            'g2': self.vehicle._is_colliding(self.goal2)
        }

    def _is_terminal(self) -> bool:
        """Determines end of episode

        Determine end of episode when:

        - The vehicle crashes or,
        - The automaton is in final state,
        - Episode duration passes an specific amount of time

        :return: a boolean indicating end of episode
        """

        return self.vehicle.crashed or self.automaton.in_final() or self.steps >= self.config['duration']

    def _reward(self, action: Action) -> float:
        # noinspection PyProtectedMember
        return super()._reward(action) + self.vehicle._is_colliding(self.goal2) * 1.0
