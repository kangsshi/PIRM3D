from types import SimpleNamespace

import main_runtime


def test_legacy_checkpoint_does_not_receive_new_keyword(monkeypatch):
    calls = []
    def legacy_load(path, map_location=None):
        calls.append((path, map_location))
        return {"legacy": True}
    monkeypatch.setattr(main_runtime, "torch", SimpleNamespace(load=legacy_load))
    assert main_runtime.load_checkpoint("model.pth", "cpu") == {"legacy": True}
    assert calls == [("model.pth", "cpu")]


def test_legacy_scaler_dispatch_and_disabled_autocast(monkeypatch):
    calls = []
    def legacy_scaler(enabled=True):
        calls.append(enabled)
        return "legacy scaler"
    fake = SimpleNamespace(cuda=SimpleNamespace(amp=SimpleNamespace(GradScaler=legacy_scaler)))
    monkeypatch.setattr(main_runtime, "torch", fake)
    assert main_runtime.make_scaler(False) == "legacy scaler"
    assert calls == [False]
    with main_runtime.amp_context(False):
        pass
