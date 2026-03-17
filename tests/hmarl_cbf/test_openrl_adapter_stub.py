import pytest

from hmarl_cbf.adapters import OpenRLAdapter


def test_openrl_adapter_stub() -> None:
    adapter = OpenRLAdapter(env=object(), high_policy=object(), low_policy=object())
    with pytest.raises(NotImplementedError):
        adapter.to_openrl_env()
    with pytest.raises(NotImplementedError):
        adapter.to_openrl_policy_io()
