import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("collect_perturbed_data_seed_auto_distribution.py")


def load_module():
    sys.modules.setdefault("sapien", types.SimpleNamespace(Pose=object))
    sys.modules.setdefault(
        "transforms3d",
        types.SimpleNamespace(quaternions=types.SimpleNamespace(mat2quat=lambda matrix: [1.0, 0.0, 0.0, 0.0])),
    )
    if "scipy" not in sys.modules:
        scipy_module = types.ModuleType("scipy")
        spatial_module = types.ModuleType("scipy.spatial")
        transform_module = types.ModuleType("scipy.spatial.transform")

        class FakeRotation:
            @classmethod
            def from_euler(cls, *_args, **_kwargs):
                return cls()

            def as_matrix(self):
                return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        transform_module.Rotation = FakeRotation
        spatial_module.transform = transform_module
        scipy_module.spatial = spatial_module
        sys.modules["scipy"] = scipy_module
        sys.modules["scipy.spatial"] = spatial_module
        sys.modules["scipy.spatial.transform"] = transform_module

    spec = importlib.util.spec_from_file_location("collect_parallel_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParallelSeedCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_seed_threads_is_forwarded_to_distributed_seed_worker(self):
        cli_args = Namespace(
            task_config="contact_perturb",
            source_task_config=None,
            perturb_seed=None,
            mode="seed",
            perturbation_num=3,
            perturbation_num_success=None,
            perturbation_num_fail=None,
            max_attempts=10,
            start_episode=None,
            x_range=[0.01, 0.02],
            y_range=[0.0, 0.0],
            z_range=[0.0, 0.0],
            roll_range=[0.0, 0.0],
            pitch_range=[0.0, 0.0],
            yaw_range=[0.0, 0.0],
            overwrite=False,
            dynamic_call_perturbation=False,
            seed_threads=4,
        )
        job = {
            "task_name": "pick_cube",
            "seed": 123,
            "source_episode": None,
            "source_task_config": None,
            "output_root": None,
        }

        command = self.module.build_worker_command(cli_args, job)

        self.assertIn("--seed-threads", command)
        self.assertEqual(command[command.index("--seed-threads") + 1], "4")

    def test_commit_attempt_output_moves_episode_zero_to_final_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt_dir = root / "attempt"
            final_dir = root / "final"
            source_dir = attempt_dir / "seed_123_success"
            (source_dir / "data").mkdir(parents=True)
            (source_dir / "video").mkdir()
            (source_dir / "data" / "episode0.hdf5").write_text("hdf5", encoding="utf-8")
            (source_dir / "video" / "episode0.mp4").write_text("mp4", encoding="utf-8")
            (source_dir / "scene_info.json").write_text(
                json.dumps({"episode_0": {"attempt_index": 7, "success": True}}),
                encoding="utf-8",
            )
            final_args = {"save_path": str(final_dir)}

            self.module.commit_attempt_output(
                source_save_path=str(source_dir),
                final_args=final_args,
                final_episode_idx=5,
                scene_seed=123,
                overwrite=False,
            )

            self.assertEqual((final_dir / "data" / "episode5.hdf5").read_text(encoding="utf-8"), "hdf5")
            self.assertEqual((final_dir / "video" / "episode5.mp4").read_text(encoding="utf-8"), "mp4")
            self.assertFalse((source_dir / "data" / "episode0.hdf5").exists())
            with open(final_dir / "scene_info.json", "r", encoding="utf-8") as file:
                scene_info = json.load(file)
            self.assertEqual(scene_info["episode_5"]["attempt_index"], 7)
            self.assertEqual((final_dir / "seed.txt").read_text(encoding="utf-8"), "123 123 123 123 123 123 ")

    def test_attempt_worker_command_uses_attempt_seed_and_save_flags(self):
        cli_args = Namespace(
            task_name="pick_cube",
            task_config="contact_perturb",
            seed=123,
            source_task_config="source_cfg",
            x_range=[0.01, 0.02],
            y_range=[0.0, 0.0],
            z_range=[0.0, 0.0],
            roll_range=[0.0, 0.0],
            pitch_range=[0.0, 0.0],
            yaw_range=[0.0, 0.0],
            dynamic_call_perturbation=False,
        )

        command = self.module.build_attempt_worker_command(
            cli_args,
            "/tmp/attempt-output",
            "/tmp/attempt-result.json",
            attempt_index=3,
            attempt_perturb_seed=1003,
            save_success=True,
            save_fail=False,
        )

        self.assertIn("--distributed-attempt-worker", command)
        self.assertEqual(command[command.index("--attempt-index") + 1], "3")
        self.assertEqual(command[command.index("--perturb-seed") + 1], "1003")
        self.assertIn("--attempt-save-success", command)
        self.assertNotIn("--attempt-save-fail", command)

    def test_strip_log_control_sequences_removes_color_and_cursor_codes(self):
        raw = "\x1b[A\x1b[A\x1b[93mWarning: bad sample\x1b[0m\rReplay: seed=1"

        cleaned = self.module.strip_log_control_sequences(raw)

        self.assertEqual(cleaned, "Warning: bad sample\nReplay: seed=1")

    def test_plain_output_redirector_writes_sanitized_text(self):
        class FakeProgress:
            def __init__(self):
                self.output_file = io.StringIO()
                self.messages = []

            def write(self, message):
                self.messages.append(message)

            def update_replay_status(self, message):
                self.messages.append(message)

        progress = FakeProgress()
        redirector = self.module._TqdmOutputRedirector(progress, plain_log=True)

        redirector.write("\x1b[A\x1b[93mhello\x1b[0m\n")
        redirector.flush()

        self.assertEqual(progress.messages, ["hello"])

    def test_build_log_run_dir_nests_task_config_and_timestamp_under_default_logs(self):
        cli_args = Namespace(task_config="ep2_1_object_pose_auto_dual", log_dir="logs")

        log_dir = self.module.build_log_run_dir(cli_args, now="20260709_123456")

        self.assertEqual(
            log_dir,
            self.module.ROBOTWIN_ROOT / "logs" / "ep2_1_object_pose_auto_dual" / "20260709_123456",
        )

    def test_build_log_run_dir_uses_custom_log_dir_as_base(self):
        cli_args = Namespace(task_config="cfg/name", log_dir="/tmp/custom_logs")

        log_dir = self.module.build_log_run_dir(cli_args, now="20260709_123456")

        self.assertEqual(log_dir, Path("/tmp/custom_logs") / "cfg_name" / "20260709_123456")


if __name__ == "__main__":
    unittest.main()
