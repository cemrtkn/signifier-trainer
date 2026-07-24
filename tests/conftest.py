"""Stub the optional heavy deps (peft, bitsandbytes) when they are not
installed, so the EM unit tests import in a plain local env. On the cluster /
SIF image the real packages exist, find_spec succeeds, and these stubs are
never installed."""

import importlib.machinery
import importlib.util
import sys
import types


def _install_stub(name: str, build) -> None:
    if name in sys.modules:
        return
    try:
        if importlib.util.find_spec(name) is not None:
            return  # real package present — leave it alone
    except ModuleNotFoundError:
        pass
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    build(mod)
    sys.modules[name] = mod


def _build_peft(mod):
    mod.PeftModel = type("PeftModel", (), {})
    mod.LoraConfig = type("LoraConfig", (), {})
    mod.get_peft_model = lambda *a, **k: None
    mod.prepare_model_for_kbit_training = lambda *a, **k: None


def _build_bitsandbytes(mod):
    nn = types.ModuleType("bitsandbytes.nn")
    nn.Linear4bit = type("Linear4bit", (), {})
    nn.Linear8bitLt = type("Linear8bitLt", (), {})
    mod.nn = nn
    sys.modules["bitsandbytes.nn"] = nn


_install_stub("peft", _build_peft)
_install_stub("bitsandbytes", _build_bitsandbytes)
