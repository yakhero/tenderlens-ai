"""Test runner that works with or without pytest: python run_tests.py"""
import importlib.util, sys, traceback
spec = importlib.util.spec_from_file_location("t", "tests/test_core.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fails = 0; names = [n for n in dir(m) if n.startswith("test_")]
for name in names:
    try:
        getattr(m, name)(); print("PASS", name)
    except Exception:
        fails += 1; print("FAIL", name); traceback.print_exc(limit=3)
print(f"\n{len(names)-fails}/{len(names)} passed")
sys.exit(1 if fails else 0)
