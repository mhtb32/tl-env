from typing import Dict, Tuple

# noinspection PyProtectedMember
from gym import GoalEnv
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import Landmark
from highway_env import utils


class SingleGoalEnv(AbstractEnv):
    """A continuous control environment with a goal.

        The vehicle is driving on a straight highway and must reach goal area, without colliding with other vehicles.
    """
    def default_config(self) -> Dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ['x', 'y', 'vx', 'vy'],
                "clip": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 5,  # [Hz]
            "duration": 20,  # [s]
            "lanes_count": 4,
            "initial_spacing": 1,
            "goal_position": [140, 12],
            "vehicle_init": None
        })
        return config

    def reset(self) -> np.ndarray:
        self._create_road()
        self._create_vehicle()
        self.steps = 0
        return super().reset()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        return super().step(action)

    def _create_road(self) -> None:
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random)
        self.goal = Landmark(self.road, self.config["goal_position"], heading=0)
        self.road.objects.append(self.goal)

    def _create_vehicle(self) -> None:
        if self.config["vehicle_init"]:
            # noinspection PyUnresolvedReferences
            self.vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road,
                self.road.network.get_closest_lane_index(self.config["vehicle_init"]["position"]),
                self.config["vehicle_init"]["position"][0],
                self.road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1]) if
                self.config["vehicle_init"]["speed"] is None else
                self.config["vehicle_init"]["speed"]
            )
            if self.config["vehicle_init"]["heading"] is None:
                # randomize initial heading
                self.vehicle.heading += self.np_random.uniform(-np.pi / 12, np.pi / 12)
            else:
                self.vehicle.heading = self.config["vehicle_init"]["heading"]
        else:
            # noinspection PyUnresolvedReferences
            self.vehicle = self.action_type.vehicle_class.create_random(self.road,
                                                                        spacing=self.config['initial_spacing'])

        self.road.vehicles.append(self.vehicle)

    def _is_terminal(self) -> bool:
        """Determines end of episode

        Determine end of episode when the vehicle reaches the goal

        :return: a boolean indicating end of episode
        """
        return self.goal.hit or self.steps >= self.config['duration']

    def _reward(self, action: Action) -> float:
        return -np.linalg.norm(self.vehicle.position - self.goal.position)

    def _cost(self, action: Action) -> float:
        pass


class SingleGoalIDMEnv(SingleGoalEnv):
    """A continuous control environment with a goal and some adversary vehicles.

        The vehicle must reach the goal while avoiding other vehicles.
    """
    def default_config(self) -> Dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 4,
                "features": ['x', 'y', 'vx', 'vy'],
                "clip": False
            },
            "vehicles_count": 10,
            "initial_spacing": 1,
            "duration": 25,
            "goal_position": [160, 12]
        })
        return config

    def _create_vehicle(self) -> None:
        super()._create_vehicle()

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            # noinspection PyUnresolvedReferences
            self.road.vehicles.append(vehicles_type.create_random(self.road))

    def _is_terminal(self) -> bool:
        """Determines end of episode

        Determine end of episode when:

        - The vehicle reaches the goal or,
        - The vehicle crashes or,
        - Episode duration passes an specific amount of time

        :return: a boolean indicating end of episode
        """
        # noinspection PyProtectedMember
        return self.vehicle._is_colliding(self.goal) or self.vehicle.crashed or self.steps >= self.config['duration']

    def _reward(self, action: Action) -> float:
        # noinspection PyProtectedMember
        return -0.1 + self.vehicle._is_colliding(self.goal) * 1.0 + self.vehicle.crashed * -1.0 \
               + (not self.vehicle.on_road) * -0.5


class SingleGymGoalEnv(SingleGoalEnv, GoalEnv):
    """Single Goal Environment compatible with HER algorithm's implementations."""
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
        obs = self.observation_type.observe()
        return self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
