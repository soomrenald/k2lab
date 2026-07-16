from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from k2_region_lab.processes import find_owned_k2_workers, is_k2_worker_command


class ProcessDiscoveryTests(unittest.TestCase):
    def test_worker_command_matching_is_narrow(self) -> None:
        self.assertTrue(
            is_k2_worker_command(
                ("python", "-m", "k2_region_lab.worker.entrypoint")
            )
        )
        self.assertTrue(
            is_k2_worker_command(
                ("python", "/checkout/k2_region_lab/worker/entrypoint.py")
            )
        )
        self.assertFalse(is_k2_worker_command(("python", "main.py", "--directml")))
        self.assertFalse(is_k2_worker_command(("python", "comfyui/main.py")))

    def test_discovery_only_returns_current_users_k2_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory)
            self._write_process(
                proc_root,
                8123,
                os.getuid(),
                100,
                ("python", "-m", "k2_region_lab.worker.entrypoint"),
            )
            self._write_process(
                proc_root,
                8124,
                os.getuid(),
                100,
                ("python", "comfyui/main.py"),
            )
            self._write_process(
                proc_root,
                8125,
                os.getuid() + 1,
                100,
                ("python", "-m", "k2_region_lab.worker.entrypoint"),
            )

            workers = find_owned_k2_workers(proc_root)

        self.assertEqual([worker.pid for worker in workers], [8123])
        self.assertEqual(workers[0].parent_pid, 100)

    @staticmethod
    def _write_process(
        root: Path,
        pid: int,
        uid: int,
        parent_pid: int,
        command: tuple[str, ...],
    ) -> None:
        process = root / str(pid)
        process.mkdir()
        (process / "status").write_text(
            f"Name:\tpython\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
            f"PPid:\t{parent_pid}\n",
            encoding="utf-8",
        )
        (process / "cmdline").write_bytes(
            b"\0".join(item.encode("utf-8") for item in command) + b"\0"
        )


if __name__ == "__main__":
    unittest.main()
