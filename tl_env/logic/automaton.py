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
        # prevents making anything but empty graph
        super().__init__()
        self.input_alphabet = set()

    def add_state(self, state: State, type_: Optional[str] = None) -> None:
        """Adds new state to automaton

        :param state: an integer or string denoting the state
        :param type_: ('init', 'final'), to mark initial or final states
        """
        if type_ is not None:
            if type_ in ('init', 'final'):
                self.add_node(state, type=type_)
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
        # add 'label' attribute for graphviz drawing
        _transitions = [(a, b, {'label': x}) for a, b, x in transitions]
        self.add_edges_from(_transitions)

    def draw(self) -> None:
        """Draws automaton using graphviz"""
        agraph = nx.nx_agraph.to_agraph(self)
        agraph.node_attr.update(shape='circle')
        # change shape of the final states
        for state, info in self.nodes.items():
            if info and info['type'] == 'final':
                node = agraph.get_node(state)
                node.attr.update(shape='doublecircle')
        agraph.layout(prog='dot')
        agraph.draw(str(Path.cwd() / 'automaton.pdf'))
