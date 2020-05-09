import pytest

from tl_env.logic.automaton import Automaton, TransitionError


def test_add_state():
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


def test_add_state_from():
    a = Automaton()

    a.add_state_from([1, 2])  # add normal states
    a.add_state_from([(3, {'type': 'final'})])
    assert a.states <= {1, 2, 3}
    assert a.states[3] == {'type': 'final'}

    # check whether key error is raised
    with pytest.raises(KeyError):
        a.add_state_from([(4, {'typo': 'final'})])


def test_step():
    a = Automaton()
    alphabet = {'x': True, 'y': False}

    a.add_transition_from([('q1', 'q2', 'x'), ('q1', 'q3', 'y'), ('q2', 'q3', 'y')])
    # initial state is unknown, so we expect throwing an error
    with pytest.raises(TransitionError):
        a.step(alphabet)

    a.add_state_from([('q1', {'type': 'init'}), 'q2', ('q3', {'type': 'final'})])

    a.step(alphabet)
    assert a.cur_state == 'q2'

    # should not allow simultaneous active alphabet
    alphabet = {'x': True, 'y': True}
    with pytest.raises(TransitionError):
        a.step(alphabet)

    alphabet = {'x': False, 'y': True}
    a.step(alphabet)
    assert a.cur_state == 'q3'
    # q3 has no successors, so the state must remain q3
    a.step(alphabet)
    assert a.cur_state == 'q3'
    assert a.in_final()


def test_drawing():
    a = Automaton()

    a.add_state('q1')
    a.add_state('q2', type_='final')
    a.add_transition_from([('q1', 'q2', 'x'), ('q2', 'q1', 'y')])

    a.draw()
