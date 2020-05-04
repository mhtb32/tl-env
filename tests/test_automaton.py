import pytest

from tl_env.logic.automaton import Automaton


def test_node_addition():
    a = Automaton()

    a.add_state(1)  # add int node
    a.add_state('q1')  # add str node
    assert a.states <= {1, 'q1'}

    a.clear()
    a.add_state(1, type_='init')
    a.add_state(2, type_='final')
    assert a.states[1] == {'type': 'init'}

    # check whether value error is raised
    with pytest.raises(ValueError):
        a.add_state(3, type_='normal')


def test_drawing():
    a = Automaton()

    a.add_state('q1')
    a.add_state('q2', type_='final')
    a.add_transition_from([('q1', 'q2', 'x'), ('q2', 'q1', 'y')])

    a.draw()
