from __future__ import annotations

import inspect

import tests.test_core as test_core
import tests.test_full as test_full


def main() -> None:
    failures: list[str] = []
    for module in (test_core, test_full):
        for name, fn in sorted(vars(module).items()):
            if name.startswith("test_") and callable(fn):
                qualified = f"{module.__name__}.{name}"
                try:
                    fn()
                    print(f"PASS {qualified}")
                except Exception as exc:
                    failures.append(qualified)
                    print(f"FAIL {qualified}: {exc}")
                    print(inspect.trace()[-1].code_context)
    if failures:
        raise SystemExit(f"{len(failures)} tests failed: {', '.join(failures)}")
    print("All tests passed")


if __name__ == "__main__":
    main()
