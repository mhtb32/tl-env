from pathlib import Path
from typing import Optional, Union, Iterable, Tuple, Dict

import networkx as nx

State = Union[int, str]
TypedState = Tuple[State, Dict]
Transition = Tuple[State, State, str]


class Automaton(nx.DiGraph):
    """A simple automaton.
        LTL formulas are translated to an Automaton instance.
    """
    def __init__(self):
        super().__init__()
        self.input_alphabet = {}
        # For now, we only allow one initial and one final state
        self.initial_state = None
        self.final_state = None
        self.cur_state = None

    def add_state(self, state: State, type_: Optional[str] = None) -> None:
        """Adds new state to automaton

        :param state: an integer or string denoting the state
        :param type_: ('init', 'final'), to mark initial or final states
        """
        if type_:
            if type_ == 'init':
                self.add_node(state, type=type_)
                self.cur_state = self.initial_state = state
            elif type_ == 'final':
                self.add_node(state, type=type_)
                self.final_state = state
            else:
                raise ValueError("node type just can be 'init' or 'final'")
        else:
            self.add_node(state)

    def add_state_from(self, states: Iterable[Union[State, TypedState]]) -> None:
        """Adds states from a container.

        example
        -------
        You can add all states at once like::

            A = Automaton()
            A.add_state_from([1, (2, {'type': 'final'})])

        :param states: an iterable container of states or typed states
        """
        for state in states:
            if isinstance(state, (int, str)):
                self.add_state(state)
            elif isinstance(state, tuple):
                try:
                    self.add_state(state[0], type_=state[1]['type'])
                except KeyError:
                    print("Dictionary key must be 'type'\n")
                    raise

    # making an alias for nodes property
    states = nx.DiGraph.nodes

    def add_transition_from(self, transitions: Iterable[Transition]) -> None:
        """Takes transitions as an iterable container of transition tuples

        :param transitions: an iterable container of transition tuples like ('q1', 'q2', 'x')
        """
        for a, b, x in transitions:
            # Graphviz interprets label keyword as edge label
            self.add_edge(a, b, label=x)
            # add transition label to alphabet
            self.input_alphabet[x] = False

    def step(self, input_: Dict) -> None:
        """Determines next state of automaton.

        Determines next state of automaton based on input alphabet status(boolean).
        An example of input alphabet dictionary::

            input_ = {
                'g1': True,
                'g2': False
            }

        :param input_: a dictionary containing input alphabet and their status(true, false)
        """
        if self.cur_state is None:
            raise TransitionError("Current state is unknown. You should provide at least one state with type 'init'.")

        self.input_alphabet = input_
        active_alphabet = []

        for alphabet, status in self.input_alphabet.items():
            if status:
                active_alphabet.append(alphabet)
        if len(active_alphabet) > 1:
            raise TransitionError("Currently more than one active alphabet is not supported")
        if not active_alphabet:
            # return if there is no active alphabet
            return

        for s in self.successors(self.cur_state):
            if self.get_edge_data(self.cur_state, s)['label'] == active_alphabet[0]:
                self.cur_state = s

    def in_final(self) -> bool:
        """Returns true if current state is a final state

        :return: a boolean indicating whether automaton is in final state or not.
        """
        return self.cur_state == self.final_state

    def draw(self) -> None:
        """Draws automaton using graphviz"""
        agraph = nx.nx_agraph.to_agraph(self)
        agraph.node_attr.update(shape='circle')
        # change shape of the final state
        final_node = agraph.get_node(self.final_state)
        final_node.attr.update(shape='doublecircle')
        agraph.layout(prog='dot')
        agraph.draw(str(Path.cwd() / 'automaton.pdf'))


class TransitionError(Exception):
    """Raised when a transition in not possible from current state of automaton"""
    pass
