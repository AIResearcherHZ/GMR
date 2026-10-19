import importlib.util
import os
import platform
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, platform.machine(), "taks_driver.abi3.so")

_taks_driver = sys.modules.get("taks_driver")
if _taks_driver is None:
    if not os.path.exists(_SO):
        raise ImportError(f"缺少 {_SO}\n在本机执行 backend/driver_rust/build.sh 生成")
    _spec = importlib.util.spec_from_file_location("taks_driver", _SO)
    assert _spec is not None and _spec.loader is not None
    _taks_driver = importlib.util.module_from_spec(_spec)
    sys.modules["taks_driver"] = _taks_driver
    _spec.loader.exec_module(_taks_driver)

sys.modules[f"{__name__}.taks_driver"] = _taks_driver
taks_driver = _taks_driver

__all__ = ["_taks_driver", "taks_driver"]
for _name in dir(_taks_driver):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_taks_driver, _name)
        __all__.append(_name)
