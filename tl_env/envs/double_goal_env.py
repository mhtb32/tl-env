from typing import Tuple, Dict

import numpy as np
from highway_env.envs.common.abstract import AbstractEnv, Action, Observation
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import Landmark

from tl_env.logic.automaton import Automaton


class DoubleGoalEnv(AbstractEnv):
    """A continuous control environment with two goals.

        The vehicle is driving on a straight highway and must reach two goal areas in a specific order, without
        colliding with other vehicles. The rewards are given based on a temporal logic formula(which specifies the
        ordering).
    """
    MAX_REWARD = 2 / (Vehicle.WIDTH + Landmark.WIDTH)
    TASK_COMPLETION_REWARD = 10

    def __init__(self, config: Dict = None) -> None:
        self.automaton = None
        super().__init__(config)

    def default_config(self) -> Dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "Continuous"
            },
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "lanes_count": 4
        })
        return config

    def reset(self) -> np.ndarray:
        self._create_automaton()
        self._create_road()
        self._create_vehicle()
        return super().reset()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        # reset goal hit status every step
        self.goal1.hit = self.goal2.hit = False
        obs, reward, terminal, info = super().step(action)
        info.update(self._goal_achievement())
        # step the automaton with newly updated info about reaching goals
        self.automaton.step(self._goal_achievement())
        # update reward and terminal with new automaton state
        reward = self._reward(action)
        terminal = self._is_terminal()
        return obs, reward, terminal, info

    def _create_automaton(self) -> None:
        self.automaton = Automaton()
        self.automaton.add_state_from([('q0', {'type': 'init'}),
                                       'q1',
                                       ('q2', {'type': 'final'})])
        self.automaton.add_transition_from([('q0', 'q1', 'g1'),
                                            ('q1', 'q2', 'g2')])

    def _create_road(self) -> None:
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random)

        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal1 = Landmark(self.road, [80, 12], heading=lane.heading)
        self.goal2 = Landmark(self.road, [160, 0], heading=lane.heading)
        self.road.objects.extend((self.goal1, self.goal2))

    def _create_vehicle(self) -> None:
        self.vehicle = Vehicle.create_random(self.road)
        self.road.vehicles.append(self.vehicle)

    def _goal_achievement(self) -> Dict:
        return {
            'g1': self.goal1.hit,
            'g2': self.goal2.hit
        }

    def _is_terminal(self) -> bool:
        """Determines end of episode

        Determine end of episode when the vehicle:
            - Reaches an accepting state
            - Crashes with another vehicle
            - Or a specific amount of time is passed

        :return: a boolean indicating end of episode
        """
        return self.vehicle.crashed or self.automaton.in_final()

    def _reward(self, action: Action) -> float:
        if self.automaton.cur_state == 'q0':
            d = np.linalg.norm(self.vehicle.position - self.goal1.position)
            return min(1 / d, self.MAX_REWARD)
        elif self.automaton.cur_state == 'q1':
            d = np.linalg.norm(self.vehicle.position - self.goal2.position)
            return min(1 / d, self.MAX_REWARD)
        elif self.automaton.cur_state == 'q2':
            return self.TASK_COMPLETION_REWARD

    def _cost(self, action: Action) -> float:
        pass
