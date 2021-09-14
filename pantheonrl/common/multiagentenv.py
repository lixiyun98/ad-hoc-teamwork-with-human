from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

import gym
import numpy as np

from .agents import Agent


class PlayerException(Exception):
    """ Raise when players in the environment are incorrectly set """


class MultiAgentEnv(gym.Env, ABC):
    """
    Base class for all Multi-agent environments.

    :param ego_ind: The player that the ego represents
    :param n_players: The number of players in the game
    :param partners: Lists of agents to choose from for the partner players
    """

    def __init__(self,
                 ego_ind: int = 0,
                 n_players: int = 2,
                 partners: Optional[List[List[Agent]]] = None):
        self.ego_ind = ego_ind
        self.n_players = n_players
        if partners is not None:
            if len(partners) != n_players - 1:
                raise PlayerException(
                    "The number of partners needs to equal the number \
                    of non-ego players")

            for plist in partners:
                if not isinstance(plist, list) or not plist:
                    raise PlayerException(
                        "Sublist for each partner must be nonempty list")

        self.partners = partners or [[]] * (n_players - 1)
        self.partnerids = [0] * (n_players - 1)

        self._players: Tuple[int, ...] = tuple()
        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()

        self.should_update = [False] * (self.n_players - 1)
        self.total_rews = [0] * (self.n_players)
        self.ego_moved = False

    def _get_partner_num(self, player_num: int) -> int:
        if player_num == self.ego_ind:
            raise PlayerException(
                "Ego agent is not set by the environment")
        elif player_num > self.ego_ind:
            return player_num - 1
        return player_num

    def add_partner_agent(self, agent: Agent, player_num: int = 1) -> None:
        """
        Add agent to the list of potential partner agents

        :param agent: Agent to add
        """
        self.partners[self._get_partner_num(player_num)].append(agent)

    def set_partnerid(self, agent_id: int, player_num: int = 1) -> None:
        """
        Set the current partner agent to use

        :param agent_id: agent_id to use as current partner
        """
        partner_num = self._get_partner_num(player_num)
        assert(agent_id >= 0 and agent_id < len(self.partners[partner_num]))
        self.partnerids[partner_num] = agent_id

    def resample_partner(self) -> None:
        """ Resample the partner agent used """
        self.partnerids = [np.random.randint(len(plist))
                           for plist in self.partners]

    def _get_actions(self, players, obs, ego_act=None):
        actions = []
        for player, ob in zip(players, obs):
            if player == self.ego_ind:
                actions.append(ego_act)
            else:
                p = self._get_partner_num(player)
                agent = self.partners[p][self.partnerids[p]]
                actions.append(agent.get_action(ob))
                if not self.should_update[p]:
                    agent.update(self.total_rews[player], False)
                self.should_update[p] = True
        return np.array(actions)

    def _update_players(self, rews, done):
        for i in range(self.n_players - 1):
            nextrew = rews[i + (0 if i < self.ego_ind else 1)]
            if self.should_update[i]:
                self.partners[i][self.partnerids[i]].update(nextrew, done)

        for i in range(self.n_players):
            self.total_rews[i] += rews[i]

    def step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], float, bool, Dict]:
        """
        Run one timestep from the perspective of the ego-agent. This involves
        calling the ego_step function and the alt_step function to get to the
        next observation of the ego agent.

        Accepts the ego-agent's action and returns a tuple of (observation,
        reward, done, info) from the perspective of the ego agent.

        :param action: An action provided by the ego-agent.

        :returns:
            observation: Ego-agent's next observation
            reward: Amount of reward returned after previous action
            done: Whether the episode has ended (need to call reset() if True)
            info: Extra information about the environment
        """
        ego_rew = 0.0

        while True:
            acts = self._get_actions(self._players, self._obs, action)
            self._players, self._obs, rews, done, info = self.n_step(acts)
            info['_partnerid'] = self.partnerids

            self._update_players(rews, done)

            ego_rew += rews[self.ego_ind] if self.ego_moved \
                else self.total_rews[self.ego_ind]

            self.ego_moved = True

            if done:
                return None, ego_rew, done, info

            if self.ego_ind in self._players:
                break

        ego_obs = self._obs[self._players.index(self.ego_ind)]

        return ego_obs, ego_rew, done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        self.resample_partner()
        self._players, self._obs = self.n_reset()
        self.should_update = [False] * (self.n_players - 1)
        self.total_rews = [0] * self.n_players
        self.ego_moved = False

        while self.ego_ind not in self._players:
            acts = self._get_actions(self._players, self._obs)
            self._players, self._obs, rews, done, _ = self.n_step(acts)

            if done:
                raise PlayerException("Game ended before ego moved")

            self._update_players(rews, done)

        ego_obs = self._obs[self._players.index(self.ego_ind)]

        assert ego_obs is not None

        return ego_obs

    @abstractmethod
    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[np.ndarray], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        """
        Perform the actions specified by the agents that will move. This
        function returns a tuple of (next agents, observations, both rewards,
        done, info).

        This function is called by the `step` function.

        :param actions: List of action provided agents that are acting on this
        step.

        :returns:
            agents: Tuple representing the agents to call for the next actions
            observations: Tuple representing the next observations (ego, alt)
            rewards: Tuple representing the rewards of all agents
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
        """
        Reset the environment and return which agents will move first along
        with their initial observations.

        This function is called by the `reset` function.

        :returns:
            agents: Tuple representing the agents that will move first
            observations: Tuple representing the observations of both agents
        """


class TurnBasedEnv(MultiAgentEnv, ABC):
    """
    Base class for all 2-player turn-based games.

    In turn-based games, players take turns receiving observations and making
    actions.

    :param probegostart: Probability that the ego agent gets the first turn
    :param partners: List of policies to choose from for the partner agent
    """

    def __init__(self,
                 probegostart: float = 0.5,
                 partners: Optional[List[Agent]] = None):
        partners = [partners] if partners else None
        super(TurnBasedEnv, self).__init__(
            ego_ind=0, n_players=2, partners=partners)
        self.probegostart = probegostart
        self.ego_next = True

    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[np.ndarray], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        agents = (1 if self.ego_next else 0,)
        obs, rews, done, info = self.ego_step(actions[0]) if self.ego_next \
            else self.alt_step(actions[0])

        self.ego_next = not self.ego_next

        return agents, (obs,), rews, done, info

    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
        self.ego_next = (np.random.rand() < self.probegostart)

        return (0 if self.ego_next else 1,), (self.multi_reset(self.ego_next),)

    @abstractmethod
    def ego_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        """
        Perform the ego-agent's action and return a tuple of (partner's
        observation, both rewards, done, info).

        This function is called by the `step` function along with alt-step.

        :param action: An action provided by the ego-agent.

        :returns:
            partner observation: Partner's next observation
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def alt_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        """
        Perform the partner's action and return a tuple of (ego's observation,
        both rewards, done, info).

        This function is called by the `step` function along with ego-step.

        :param action: An action provided by the partner.

        :returns:
            ego observation: Ego-agent's next observation
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def multi_reset(self, egofirst: bool) -> np.ndarray:
        """
        Reset the environment and give the observation of the starting agent
        (based on the value of `egofirst`).

        This function is called by the `reset` function.

        :param egofirst: True if the ego has the first turn, False otherwise
        :returns: The observation for the starting agent (ego if `egofirst` is
            True, and the partner's observation otherwise)
        """


class SimultaneousEnv(MultiAgentEnv, ABC):
    """
    Base class for all 2-player simultaneous games.

    :param partners: List of policies to choose from for the partner agent
    """

    def __init__(self, partners: Optional[List[Agent]] = None):
        partners = [partners] if partners else None
        super(SimultaneousEnv, self).__init__(
            ego_ind=0, n_players=2, partners=partners)

    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[np.ndarray], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        return ((0, 1),) + self.multi_step(actions[0], actions[1])

    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
        return (0, 1), self.multi_reset()

    @abstractmethod
    def multi_step(
                    self,
                    ego_action: np.ndarray,
                    alt_action: np.ndarray
                ) -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                           Tuple[float, float], bool, Dict]:
        """
        Perform the ego-agent's and partner's actions. This function returns a
        tuple of (observations, both rewards, done, info).

        This function is called by the `step` function.

        :param ego_action: An action provided by the ego-agent.
        :param alt_action: An action provided by the partner.

        :returns:
            observations: Tuple representing the next observations (ego, alt)
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def multi_reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment and give the observation of both agents.

        This function is called by the `reset` function.

        :returns: The observations of both agents
        """