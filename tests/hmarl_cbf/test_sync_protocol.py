from hmarl_cbf.control import SyncCoordinator


def test_sync_switch_by_all_terminated() -> None:
    coordinator = SyncCoordinator(num_agents=3, t_sync_max=10)
    result = coordinator.step({0: True, 1: True, 2: True})
    assert result.sync_switch is True
    assert coordinator.k == 1


def test_sync_switch_by_timeout() -> None:
    coordinator = SyncCoordinator(num_agents=2, t_sync_max=3)
    coordinator.step({0: False, 1: False})
    coordinator.step({0: False, 1: False})
    result = coordinator.step({0: False, 1: False})
    assert result.sync_switch is True
    assert coordinator.k == 1
