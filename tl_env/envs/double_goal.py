from typing import Dict, Tuple

from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import Action
from highway_env.road.objects import Landmark

from tl_env.envs.single_goal import SingleGoalIDMEnv


class DoubleGoalEnv(SingleGoalIDMEnv):
    """A continuous control environment with two goals and some adversary vehicles.

    The vehicle must reach the goals while avoiding other vehicles.
    """
    def default_config(self) -> Dict:
        config = super().default_config()
        config.update(
            {
                "duration": 50,
                "goal2_position": [260, 4]
            }
        )
        return config

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        obs, reward, terminal, info = super().step(action)
        goal_achievement = self._goal_achievement()
        # calculate mid-done flag
        mid_done = self.vehicle.crashed or goal_achievement['g1'] or self.steps >= 25
        # update info and terminal with new automaton state
        info.update(dict(mid_done=mid_done))
        terminal = self._is_terminal()
        return obs, reward, terminal, info

    def _create_road(self) -> None:
        super()._create_road()
        self.goal2 = Landmark(self.road, self.config["goal2_position"], heading=0)
        self.road.objects.append(self.goal2)

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
        - Reaches the second goal or,
        - Episode duration passes an specific amount of time

        :return: a boolean indicating end of episode
        """

        return self.vehicle.crashed or self._goal_achievement()['g2'] or self.steps >= self.config['duration']

    def _reward(self, action: Action) -> float:
        # noinspection PyProtectedMember
        return super()._reward(action) + self.vehicle._is_colliding(self.goal2) * 1.0
