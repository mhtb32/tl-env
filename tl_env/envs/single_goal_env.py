from typing import Dict

# noinspection PyProtectedMember
from gym import GoalEnv
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv, Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import Landmark


class SingleGoalEnv(AbstractEnv):
    """A continuous control environment with a goal.

        The vehicle is driving on a straight highway and must reach goal area, without colliding with other vehicles.
    """

    def __init__(self, config: Dict = None) -> None:
        self.automaton = None
        super().__init__(config)

    def default_config(self) -> Dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h']
            },
            "action": {
                "type": "Continuous"
            },
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 5,  # [Hz]
            "lanes_count": 4
        })
        return config

    def reset(self) -> np.ndarray:
        self._create_road()
        self._create_vehicle()
        return super().reset()

    def _create_road(self) -> None:
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random)

        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(self.road, [80, 12], heading=lane.heading)
        self.road.objects.append(self.goal)

    def _create_vehicle(self) -> None:
        self.vehicle = Vehicle.create_random(self.road)
        self.road.vehicles.append(self.vehicle)

    def _is_terminal(self) -> bool:
        """Determines end of episode

        Determine end of episode when the vehicle:
            - Reaches an accepting state
            - Crashes with another vehicle
            - Or leaves the road

        :return: a boolean indicating end of episode
        """
        return self.vehicle.crashed or self.goal.hit or not self.vehicle.on_road

    def _reward(self, action: Action) -> float:
        return -np.linalg.norm(self.vehicle.position - self.goal.position)

    def _cost(self, action: Action) -> float:
        pass


class SingleGymGoalEnv(SingleGoalEnv, GoalEnv):
    """
        Single Goal Environment inheriting from gym GoalEnv to be compatible with HER algorithm's implementations.
    """
    def default_config(self) -> Dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "vehicles_count": 1,
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [1, 1, 1, 1, 1, 1]
            },
        })
        return config

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict) -> float:
        """Rewards the agent based on proximity to goal

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :return: the corresponding reward
        """
        return -np.linalg.norm(achieved_goal - desired_goal)

    def _reward(self, action: Action) -> float:
        obs = self.observation.observe()
        return self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
