import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from argparse import ArgumentParser
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROBOTWIN_ROOT)
sys.path.append(str(ROBOTWIN_ROOT))

import importlib

import numpy as np
import sapien
import transforms3d as t3d
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

CONFIGS_PATH = str(ROBOTWIN_ROOT / "task_config")
DEFAULT_DATASET_ROOT = (
    "/data/wangbowen/PHD_Research/02_Action_Generalization/Action_Hallucination_Verification/"
    "robotwin_consistency/database/datasets/robotwin_480_640"
)

_ORIGINAL_ACTOR_GET_POINT = None
_ORIGINAL_ARTICULATION_GET_POINT = None
_ORIGINAL_TAKE_PICTURE = None
_CONTACT_PERTURBATION_SAMPLER = None
_REPLAY_PROGRESS = None


class ContactPointPerturbationSampler:
    def __init__(self, ranges, seed=None, fixed_episode_perturbation=True, max_resample=100):
        self.ranges = ranges
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fixed_episode_perturbation = fixed_episode_perturbation
        self.max_resample = max_resample
        self.episode_info = {}
        self._episode_cache = {}
        self._call_index = 0
        self._validate_ranges()

    def _validate_ranges(self):
        has_nonzero = False
        for values in self.ranges.values():
            lo, hi = float(values[0]), float(values[1])
            if lo > hi:
                raise SystemExit(f"Invalid perturbation range [{lo}, {hi}], min must be <= max")
            if lo != 0.0 or hi != 0.0:
                has_nonzero = True
        if not has_nonzero:
            raise SystemExit("At least one perturbation range must contain a non-zero value")

    def begin_episode(self, meta, phase, reset=False):
        if reset:
            self._episode_cache = {}
            self._call_index = 0
            self.episode_info = {
                "seed": meta.get("scene_seed"),
                "source_mode": meta.get("source_mode"),
                "source_task": meta.get("source_task"),
                "source_task_config": meta.get("source_task_config"),
                "source_episode": meta.get("source_episode"),
                "perturbation_index_for_source": meta.get("perturbation_index_for_source"),
                "attempt_index": meta.get("attempt_index"),
                "perturb_seed": self.seed,
                "fixed_episode_perturbation": self.fixed_episode_perturbation,
                "contact_perturbation_ranges": self.serializable_ranges(),
                "contact_perturbation_samples": [],
            }
        self.episode_info["phase"] = phase

    def serializable_ranges(self):
        return {
            "x": list(self.ranges["x"]),
            "y": list(self.ranges["y"]),
            "z": list(self.ranges["z"]),
            "roll_deg": list(self.ranges["roll_deg"]),
            "pitch_deg": list(self.ranges["pitch_deg"]),
            "yaw_deg": list(self.ranges["yaw_deg"]),
        }

    def _sample_xyz_rpy(self):
        for _ in range(self.max_resample):
            dx = self.rng.uniform(*self.ranges["x"])
            dy = self.rng.uniform(*self.ranges["y"])
            dz = self.rng.uniform(*self.ranges["z"])
            roll_deg = self.rng.uniform(*self.ranges["roll_deg"])
            pitch_deg = self.rng.uniform(*self.ranges["pitch_deg"])
            yaw_deg = self.rng.uniform(*self.ranges["yaw_deg"])
            xyz_rpy_deg = np.array([dx, dy, dz, roll_deg, pitch_deg, yaw_deg], dtype=np.float64)
            if not np.allclose(xyz_rpy_deg, 0.0):
                xyz_rpy = xyz_rpy_deg.copy()
                xyz_rpy[3:] = np.deg2rad(xyz_rpy[3:])
                return xyz_rpy, xyz_rpy_deg
        raise RuntimeError("Failed to sample a non-zero contact perturbation")

    def _sample_entry(self):
        xyz_rpy, xyz_rpy_deg = self._sample_xyz_rpy()
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = xyz_rpy[:3]
        matrix[:3, :3] = Rotation.from_euler("xyz", xyz_rpy[3:], degrees=False).as_matrix()
        return {
            "xyz_rpy": xyz_rpy.tolist(),
            "xyz_rpy_deg": xyz_rpy_deg.tolist(),
            "perturbation_matrix": matrix.tolist(),
        }

    def sample_for_contact(self, actor_name, actor_kind, point_idx, local_matrix, world_matrix, base_link=None):
        key = (actor_kind, actor_name, base_link, int(point_idx))
        reused = False
        if self.fixed_episode_perturbation and key in self._episode_cache:
            entry = deepcopy(self._episode_cache[key])
            reused = True
        else:
            entry = self._sample_entry()
            if self.fixed_episode_perturbation:
                self._episode_cache[key] = deepcopy(entry)

        perturbation_matrix = np.array(entry["perturbation_matrix"], dtype=np.float64)
        sample_info = {
            "actor_name": actor_name,
            "actor_kind": actor_kind,
            "point_type": "contact",
            "point_idx": int(point_idx),
            "base_link": base_link,
            "call_index": self._call_index,
            "phase": self.episode_info.get("phase"),
            "reused": reused,
            "xyz_rpy": entry["xyz_rpy"],
            "xyz_rpy_deg": entry["xyz_rpy_deg"],
            "perturbation_matrix": entry["perturbation_matrix"],
            "local_matrix": np.asarray(local_matrix).tolist(),
            "world_matrix": np.asarray(world_matrix).tolist(),
        }
        self._call_index += 1
        self.episode_info.setdefault("contact_perturbation_samples", []).append(sample_info)
        return perturbation_matrix

    def current_episode_info(self):
        return deepcopy(self.episode_info)


def _format_point_return(world_matrix, ret):
    if ret == "matrix":
        return world_matrix
    quat = t3d.quaternions.mat2quat(world_matrix[:3, :3])
    if ret == "list":
        return world_matrix[:3, 3].tolist() + quat.tolist()
    return sapien.Pose(world_matrix[:3, 3], quat)


def _get_actor_name(actor_wrapper):
    try:
        return actor_wrapper.get_name()
    except Exception:
        try:
            return actor_wrapper.actor.get_name()
        except Exception:
            return "<unknown>"


def _patched_actor_get_point(self, type, idx, ret):
    point_type = self.POINTS[type]
    actor_matrix = self.actor.get_pose().to_transformation_matrix()
    try:
        local_matrix = np.array(self.config[point_type][idx], dtype=np.float64)
    except Exception:
        return None
    local_matrix[:3, 3] *= np.array(self.config["scale"])

    world_matrix = actor_matrix @ local_matrix
    if type == "contact" and _CONTACT_PERTURBATION_SAMPLER is not None:
        perturbation_matrix = _CONTACT_PERTURBATION_SAMPLER.sample_for_contact(
            actor_name=_get_actor_name(self),
            actor_kind="Actor",
            point_idx=idx,
            local_matrix=local_matrix,
            world_matrix=world_matrix,
        )
        world_matrix = world_matrix @ perturbation_matrix

    return _format_point_return(world_matrix, ret)


def _patched_articulation_get_point(self, type, idx, ret):
    point_type = self.POINTS[type]
    local_matrix = np.array(self.config[point_type][idx]["matrix"], dtype=np.float64)
    local_matrix[:3, 3] *= self.config["scale"]

    base_link = self.config[point_type][idx]["base"]
    link = self.link_dict[base_link]
    link_matrix = link.get_pose().to_transformation_matrix()
    world_matrix = link_matrix @ local_matrix
    if type == "contact" and _CONTACT_PERTURBATION_SAMPLER is not None:
        perturbation_matrix = _CONTACT_PERTURBATION_SAMPLER.sample_for_contact(
            actor_name=_get_actor_name(self),
            actor_kind="ArticulationActor",
            point_idx=idx,
            local_matrix=local_matrix,
            world_matrix=world_matrix,
            base_link=base_link,
        )
        world_matrix = world_matrix @ perturbation_matrix

    return _format_point_return(world_matrix, ret)


def _patched_take_picture(self):
    if _REPLAY_PROGRESS is not None and getattr(self, "save_data", False):
        _REPLAY_PROGRESS.update_replay_frame(getattr(self, "ep_num", None), getattr(self, "FRAME_IDX", None))
    return _ORIGINAL_TAKE_PICTURE(self)


def install_contact_point_perturbation_patch(sampler):
    global _ORIGINAL_ACTOR_GET_POINT
    global _ORIGINAL_ARTICULATION_GET_POINT
    global _ORIGINAL_TAKE_PICTURE
    global _CONTACT_PERTURBATION_SAMPLER

    from envs.utils import actor_utils
    from envs._base_task import Base_Task

    if _ORIGINAL_ACTOR_GET_POINT is None:
        _ORIGINAL_ACTOR_GET_POINT = actor_utils.Actor.get_point
    if _ORIGINAL_ARTICULATION_GET_POINT is None:
        _ORIGINAL_ARTICULATION_GET_POINT = actor_utils.ArticulationActor.get_point
    if _ORIGINAL_TAKE_PICTURE is None:
        _ORIGINAL_TAKE_PICTURE = Base_Task._take_picture

    _CONTACT_PERTURBATION_SAMPLER = sampler
    actor_utils.Actor.get_point = _patched_actor_get_point
    actor_utils.ArticulationActor.get_point = _patched_articulation_get_point
    Base_Task._take_picture = _patched_take_picture


def uninstall_contact_point_perturbation_patch():
    global _CONTACT_PERTURBATION_SAMPLER

    from envs.utils import actor_utils
    from envs._base_task import Base_Task

    if _ORIGINAL_ACTOR_GET_POINT is not None:
        actor_utils.Actor.get_point = _ORIGINAL_ACTOR_GET_POINT
    if _ORIGINAL_ARTICULATION_GET_POINT is not None:
        actor_utils.ArticulationActor.get_point = _ORIGINAL_ARTICULATION_GET_POINT
    if _ORIGINAL_TAKE_PICTURE is not None:
        Base_Task._take_picture = _ORIGINAL_TAKE_PICTURE
    _CONTACT_PERTURBATION_SAMPLER = None


class OverallCollectionProgress:
    def __init__(self, total_tasks):
        self.output_file = sys.stdout
        self.completed_tasks = 0
        self.current = {
            "task": "-",
            "seed": "-",
            "collected": 0,
            "target": 0,
            "target_success": 0,
            "target_task_failed": 0,
            "done_success": 0,
            "done_task_failed": 0,
        }
        self.replay_total_frames = 0
        self.replay_progress = tqdm(
            total=1,
            desc="Replay",
            position=0,
            leave=True,
            dynamic_ncols=False,
            ncols=self._progress_width(),
            file=self.output_file,
            bar_format="{desc}: {percentage:3.0f}%|{bar:10}| {postfix}",
        )
        self.detail_progress = tqdm(
            total=0,
            desc="Cur",
            position=1,
            leave=True,
            dynamic_ncols=False,
            ncols=self._progress_width(),
            file=self.output_file,
            bar_format="{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} {postfix}",
        )
        self.task_progress = tqdm(
            total=total_tasks,
            desc="Tasks",
            position=2,
            leave=True,
            dynamic_ncols=False,
            ncols=self._progress_width(),
            file=self.output_file,
            bar_format="{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} {postfix}",
        )
        self.refresh()

    def _progress_width(self):
        return min(shutil.get_terminal_size((120, 20)).columns, 120)

    def _short_text(self, value, max_len=18):
        value = str(value)
        if len(value) <= max_len:
            return value
        return value[: max_len - 1] + "~"

    def start_task(self, task_name, seed="-", target=0, target_success=0, target_task_failed=0):
        self.current = {
            "task": task_name,
            "seed": seed,
            "collected": 0,
            "target": target,
            "target_success": target_success,
            "target_task_failed": target_task_failed,
            "done_success": 0,
            "done_task_failed": 0,
        }
        self.detail_progress.reset(total=target)
        self.detail_progress.n = 0
        self.refresh()

    def update_current(
        self,
        seed=None,
        collected=None,
        done_success=None,
        done_task_failed=None,
        target=None,
        target_success=None,
        target_task_failed=None,
    ):
        if seed is not None:
            self.current["seed"] = seed
        if collected is not None:
            self.current["collected"] = collected
        if done_success is not None:
            self.current["done_success"] = done_success
        if done_task_failed is not None:
            self.current["done_task_failed"] = done_task_failed
        if target is not None:
            self.current["target"] = target
            if self.detail_progress.total != target:
                self.detail_progress.total = target
        if target_success is not None:
            self.current["target_success"] = target_success
        if target_task_failed is not None:
            self.current["target_task_failed"] = target_task_failed
        self.detail_progress.n = min(self.current["collected"], self.current["target"])
        self.refresh()

    def finish_task(self):
        self.completed_tasks += 1
        self.task_progress.update(1)
        self.refresh()

    def write(self, message):
        self._ensure_clean_line()
        tqdm.write(message, file=self.output_file)

    def update_replay_status(self, message):
        self._ensure_clean_line()
        episode_idx, frame_idx = self._parse_replay_status(message)
        self.update_replay_frame(episode_idx, frame_idx)

    def update_replay_frame(self, episode_idx, frame_idx):
        self._ensure_clean_line()
        if frame_idx is not None:
            self.replay_progress.n = min(frame_idx + 1, self.replay_progress.total)
        total = self.replay_total_frames or "?"
        index_text = "?" if frame_idx is None else str(frame_idx + 1)
        if episode_idx is None:
            self.replay_progress.set_postfix_str(f"index={index_text}/{total}")
        else:
            self.replay_progress.set_postfix_str(f"episode={episode_idx} index={index_text}/{total}")
        self.replay_progress.refresh()

    def start_replay(self, episode_idx, total_frames):
        total_frames = max(int(total_frames or 0), 0)
        self.replay_total_frames = total_frames
        self.replay_progress.reset(total=total_frames or 1)
        self.replay_progress.n = 0
        self.replay_progress.set_postfix_str(f"episode={episode_idx} index=0/{total_frames or '?'}")
        self.replay_progress.refresh()

    def _parse_replay_status(self, message):
        parts = str(message).replace("=", " ").replace(",", " ").split()
        episode_idx = None
        frame_idx = None
        try:
            episode_idx = int(parts[parts.index("episode") + 1])
        except (ValueError, IndexError):
            pass
        try:
            frame_idx = int(parts[parts.index("index") + 1])
        except (ValueError, IndexError):
            pass
        return episode_idx, frame_idx

    def _ensure_clean_line(self):
        if getattr(self, "_line_dirty", False):
            self.output_file.write("\n")
            self.output_file.flush()
            self._line_dirty = False

    @contextmanager
    def redirect_output(self):
        global _REPLAY_PROGRESS
        redirector = _TqdmOutputRedirector(self)
        previous_progress = _REPLAY_PROGRESS
        _REPLAY_PROGRESS = self
        try:
            with redirect_stdout(redirector):
                yield
        finally:
            _REPLAY_PROGRESS = previous_progress
            redirector.flush()

    def refresh(self):
        c = self.current
        self._ensure_clean_line()
        self.detail_progress.set_postfix_str(
            f"Seed={c['seed']} Need_Success/Fail={c['target_success']}/{c['target_task_failed']} "
            f"Done_Success/Fail={c['done_success']}/{c['done_task_failed']}"
        )
        self.task_progress.set_postfix_str(
            f"Task={self._short_text(c['task'])} finished={self.completed_tasks}/{self.task_progress.total}"
        )
        self.replay_progress.refresh()
        self.detail_progress.refresh()
        self.task_progress.refresh()

    def close(self):
        self.replay_progress.close()
        self.detail_progress.close()
        self.task_progress.close()


class _TqdmOutputRedirector:
    def __init__(self, progress):
        self.progress = progress
        self.buffer = ""

    def write(self, text):
        if not text:
            return 0
        if "\r" in text:
            carriage_line = (self.buffer + text).replace("\r", "").strip()
            self.buffer = ""
            if carriage_line:
                self.progress.update_replay_status(carriage_line)
            return len(text)
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.progress.write(line)
        return len(text)

    def flush(self):
        if self.buffer:
            self.progress.write(self.buffer)
            self.buffer = ""
        self.progress.output_file.flush()


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        return getattr(envs_module, task_name)()
    except Exception:
        raise SystemExit("No such task")


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def read_seed_list(save_path):
    seed_file_path = os.path.join(save_path, "seed.txt")
    if not os.path.exists(seed_file_path):
        return []
    with open(seed_file_path, "r", encoding="utf-8") as file:
        return [int(seed) for seed in file.read().split()]


def write_seed(save_path, episode_idx, seed):
    os.makedirs(save_path, exist_ok=True)
    seed_path = os.path.join(save_path, "seed.txt")
    seeds = []
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as file:
            seeds = file.read().split()
    while len(seeds) <= episode_idx:
        seeds.append(str(seed))
    seeds[episode_idx] = str(seed)
    with open(seed_path, "w", encoding="utf-8") as file:
        file.write(" ".join(seeds))
        if seeds:
            file.write(" ")


def existing_episode_outputs(save_path, episode_idx):
    paths = [
        os.path.join(save_path, "data", f"episode{episode_idx}.hdf5"),
        os.path.join(save_path, "video", f"episode{episode_idx}.mp4"),
    ]
    return [path for path in paths if os.path.exists(path)]


def next_episode_index(save_path):
    episode_idx = 0
    while existing_episode_outputs(save_path, episode_idx):
        episode_idx += 1
    return episode_idx


def ensure_episode_writable(save_path, episode_idx, overwrite):
    existing_paths = existing_episode_outputs(save_path, episode_idx)
    if existing_paths and not overwrite:
        existing = ", ".join(existing_paths)
        raise SystemExit(f"Episode {episode_idx} already exists: {existing}. Use --overwrite to regenerate it.")


def close_env_safely(TASK_ENV, render_freq=0, clear_cache=False):
    try:
        TASK_ENV.close_env(clear_cache=clear_cache)
    except Exception:
        pass
    if render_freq:
        try:
            TASK_ENV.viewer.close()
        except Exception:
            pass


def remove_data_cache_safely(TASK_ENV):
    if not hasattr(TASK_ENV, "folder_path") or "cache" not in TASK_ENV.folder_path:
        return
    try:
        TASK_ENV.remove_data_cache()
    except Exception as e:
        print(f"\033[93mWarning: failed to remove data cache: {e}\033[0m")


def save_scene_info(args, episode_idx, info):
    info_file_path = os.path.join(args["save_path"], "scene_info.json")
    if not os.path.exists(info_file_path):
        with open(info_file_path, "w", encoding="utf-8") as file:
            json.dump({}, file, ensure_ascii=False)

    with open(info_file_path, "r", encoding="utf-8") as file:
        info_db = json.load(file)

    info_db[f"episode_{episode_idx}"] = info

    with open(info_file_path, "w", encoding="utf-8") as file:
        json.dump(info_db, file, ensure_ascii=False, indent=4)


def read_collected_source_keys(save_path):
    info_file_path = os.path.join(save_path, "scene_info.json")
    if not os.path.exists(info_file_path):
        return set()

    try:
        with open(info_file_path, "r", encoding="utf-8") as file:
            info_db = json.load(file)
    except Exception as e:
        print(f"\033[93mWarning: failed to read existing scene_info.json: {e}\033[0m")
        return set()

    keys = set()
    for info in info_db.values():
        if not isinstance(info, dict):
            continue
        source_task = info.get("source_task")
        source_episode = info.get("source_episode")
        perturb_idx = info.get("perturbation_index_for_source")
        if source_task is None or source_episode is None or perturb_idx is None:
            continue
        keys.add((str(source_task), int(source_episode), int(perturb_idx)))
    return keys


def finalize_hdf5_episode(TASK_ENV, args, episode_idx, clear_cache=False, require_cache=True):
    original_save_dir = getattr(TASK_ENV, "save_dir", None)
    original_ep_num = getattr(TASK_ENV, "ep_num", None)
    closed = False
    try:
        TASK_ENV.save_dir = args["save_path"]
        TASK_ENV.ep_num = episode_idx
        TASK_ENV.close_env(clear_cache=clear_cache)
        closed = True
        try:
            TASK_ENV.merge_pkl_to_hdf5_video()
        except Exception as e:
            if require_cache:
                raise
            print(f"\033[93mWarning: failed to merge episode {episode_idx}: {e}\033[0m")
        finally:
            remove_data_cache_safely(TASK_ENV)
    finally:
        if not closed:
            close_env_safely(TASK_ENV, clear_cache=clear_cache)
        if original_save_dir is not None:
            TASK_ENV.save_dir = original_save_dir
        if original_ep_num is not None:
            TASK_ENV.ep_num = original_ep_num


def save_traj_data_safely(TASK_ENV, save_path, episode_idx):
    original_save_dir = getattr(TASK_ENV, "save_dir", None)
    try:
        TASK_ENV.save_dir = save_path
        TASK_ENV.save_traj_data(episode_idx)
    finally:
        if original_save_dir is not None:
            TASK_ENV.save_dir = original_save_dir


def trajectory_frame_count(traj_data, save_freq):
    left_path = traj_data.get("left_joint_path") or []
    right_path = traj_data.get("right_joint_path") or []
    total = 0
    for idx in range(max(len(left_path), len(right_path))):
        left_steps = trajectory_result_step_count(left_path[idx] if idx < len(left_path) else None)
        right_steps = trajectory_result_step_count(right_path[idx] if idx < len(right_path) else None)
        total += saved_frame_count_for_steps(max(left_steps, right_steps), save_freq)
    return total


def saved_frame_count_for_steps(step_count, save_freq):
    if step_count <= 0 or save_freq is None:
        return 0
    save_freq = int(save_freq)
    if save_freq <= 0:
        return 0
    return 2 + ((step_count - 1) // save_freq + 1)


def trajectory_result_step_count(result):
    if not isinstance(result, dict):
        return 0
    position = result.get("position")
    shape = getattr(position, "shape", None)
    if shape:
        return int(shape[0])
    if position is not None:
        try:
            return len(position)
        except TypeError:
            pass
    try:
        return int(result.get("num_step", 0))
    except (TypeError, ValueError):
        return 0


def remove_traj_data_safely(save_path, episode_idx):
    traj_path = os.path.join(save_path, "_traj_data", f"episode{episode_idx}.pkl")
    try:
        if os.path.exists(traj_path):
            os.remove(traj_path)
    except Exception as e:
        print(f"\033[93mWarning: failed to remove temp traj data {traj_path}: {e}\033[0m")


def generate_episode_instructions(args):
    from script.utils.patch import generate_episode_instructions_from_save_path

    generate_episode_instructions_from_save_path(args)


def build_args(task_name, task_config, output_task_config=None, output_save_root=None):
    config_path = ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["source_task_config"] = task_config
    args["task_config"] = output_task_config or task_config

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        return embodiment_types[embodiment_type]["file_path"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    else:
        raise SystemExit("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name
    if output_save_root is not None:
        args["save_path"] = str(output_save_root)
    args["save_path"] = os.path.join(args["save_path"], task_name, args["task_config"])
    return args


def make_seed_output_args(args, seed):
    success_args = args.copy()
    success_args["task_config"] = os.path.join(args["task_config"], f"seed_{seed}_success")
    success_args["save_path"] = os.path.join(args["save_path"], f"seed_{seed}_success")

    fail_args = args.copy()
    fail_args["task_config"] = os.path.join(args["task_config"], f"seed_{seed}_fail")
    fail_args["save_path"] = os.path.join(args["save_path"], f"seed_{seed}_fail")
    return success_args, fail_args


def make_success_fail_output_args(args):
    success_args = args.copy()
    success_args["task_config"] = f"{args['task_config']}_success"
    success_args["save_path"] = f"{args['save_path']}_success"

    fail_args = args.copy()
    fail_args["task_config"] = f"{args['task_config']}_fail"
    fail_args["save_path"] = f"{args['save_path']}_fail"
    return success_args, fail_args


def build_episode_info(env_info, sampler, meta, success=True, failure_reason=None):
    if not isinstance(env_info, dict):
        env_info = {}
    info = deepcopy(env_info)
    info.update(sampler.current_episode_info())
    info["seed"] = meta.get("scene_seed")
    info["source_mode"] = meta.get("source_mode")
    info["source_task"] = meta.get("source_task")
    info["source_task_config"] = meta.get("source_task_config")
    info["source_episode"] = meta.get("source_episode")
    info["perturbation_index_for_source"] = meta.get("perturbation_index_for_source")
    info["attempt_index"] = meta.get("attempt_index")
    info["success"] = success
    if failure_reason is not None:
        info["failure_reason"] = str(failure_reason)
    return info


def collect_one_perturbed_episode(TASK_ENV, args, episode_idx, scene_seed, sampler, meta, overwrite=False, progress=None):
    ensure_episode_writable(args["save_path"], episode_idx, overwrite)
    plan_render_freq = args.get("render_freq", 0)
    clear_cache_freq = args.get("clear_cache_freq", 1) or 1

    plan_args = args.copy()
    plan_args["need_plan"] = True
    plan_args["save_data"] = False
    plan_args["render_freq"] = plan_render_freq
    plan_args.pop("left_joint_path", None)
    plan_args.pop("right_joint_path", None)

    sampler.begin_episode(meta, phase="plan", reset=True)
    try:
        TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=scene_seed, **plan_args)
        TASK_ENV.play_once()
        if not TASK_ENV.plan_success:
            print(f"\033[93mPlanning error: episode {episode_idx}, seed {scene_seed}\033[0m")
            return "planning_error"
        if not TASK_ENV.check_success():
            print(
                f"\033[93mTask success error during planning: episode {episode_idx}, "
                f"seed {scene_seed}; keep trajectory for failed-sample replay\033[0m"
            )
        save_traj_data_safely(TASK_ENV, args["save_path"], episode_idx)
    finally:
        close_env_safely(TASK_ENV, render_freq=plan_render_freq)

    replay_args = args.copy()
    replay_args["need_plan"] = False
    replay_args["save_data"] = True
    replay_args["render_freq"] = 0

    finalized = False
    env_info = {}
    try:
        sampler.begin_episode(meta, phase="replay", reset=False)
        TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=scene_seed, **replay_args)
        traj_data = TASK_ENV.load_tran_data(episode_idx)
        replay_args["left_joint_path"] = traj_data["left_joint_path"]
        replay_args["right_joint_path"] = traj_data["right_joint_path"]
        TASK_ENV.set_path_lst(replay_args)
        if progress is not None:
            progress.start_replay(episode_idx, trajectory_frame_count(traj_data, replay_args.get("save_freq")))

        env_info = TASK_ENV.play_once()
        if not TASK_ENV.plan_success:
            print(f"\033[93mReplay planning error: episode {episode_idx}, seed {scene_seed}\033[0m")
            close_env_safely(TASK_ENV, clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            remove_data_cache_safely(TASK_ENV)
            finalized = True
            return "planning_error"
        if not TASK_ENV.check_success():
            print(f"\033[93mTask success error: episode {episode_idx}, seed {scene_seed}\033[0m")
            close_env_safely(TASK_ENV, clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            remove_data_cache_safely(TASK_ENV)
            finalized = True
            return "task_error"

        save_scene_info(args, episode_idx, build_episode_info(env_info, sampler, meta, success=True))
        finalize_hdf5_episode(
            TASK_ENV,
            args,
            episode_idx,
            clear_cache=((episode_idx + 1) % clear_cache_freq == 0),
            require_cache=True,
        )
        finalized = True
        write_seed(args["save_path"], episode_idx, scene_seed)
        return "success"
    except Exception as e:
        if not finalized:
            try:
                save_scene_info(args, episode_idx, build_episode_info(env_info, sampler, meta, success=False, failure_reason=e))
            except Exception:
                pass
        raise
    finally:
        if not finalized:
            close_env_safely(TASK_ENV, clear_cache=((episode_idx + 1) % clear_cache_freq == 0))


def collect_one_seed_perturbed_episode(
    TASK_ENV,
    success_args,
    fail_args,
    success_episode_idx,
    fail_episode_idx,
    plan_episode_idx,
    plan_save_path,
    scene_seed,
    sampler,
    meta,
    overwrite=False,
    save_success=True,
    save_fail=True,
    progress=None,
):
    if save_success:
        ensure_episode_writable(success_args["save_path"], success_episode_idx, overwrite)
    if save_fail:
        ensure_episode_writable(fail_args["save_path"], fail_episode_idx, overwrite=False)
    plan_render_freq = success_args.get("render_freq", 0)
    clear_cache_freq = success_args.get("clear_cache_freq", 1) or 1

    plan_args = success_args.copy()
    plan_args["need_plan"] = True
    plan_args["save_data"] = False
    plan_args["render_freq"] = plan_render_freq
    plan_args.pop("left_joint_path", None)
    plan_args.pop("right_joint_path", None)

    sampler.begin_episode(meta, phase="plan", reset=True)
    try:
        TASK_ENV.setup_demo(now_ep_num=plan_episode_idx, seed=scene_seed, **plan_args)
        TASK_ENV.play_once()
        if not TASK_ENV.plan_success:
            print(
                f"\033[93mPlanning error: plan episode {plan_episode_idx}, seed {scene_seed}, "
                f"attempt {meta.get('attempt_index')}\033[0m"
            )
            return "planning_error"
        save_traj_data_safely(TASK_ENV, plan_save_path, plan_episode_idx)
    finally:
        close_env_safely(TASK_ENV, render_freq=plan_render_freq)

    replay_args = success_args.copy()
    replay_args["need_plan"] = False
    replay_args["save_data"] = True
    replay_args["render_freq"] = 0

    finalized = False
    env_info = {}
    play_error = None
    check_error = None
    try:
        sampler.begin_episode(meta, phase="replay", reset=False)
        output_episode_idx = success_episode_idx if save_success else fail_episode_idx
        TASK_ENV.setup_demo(now_ep_num=output_episode_idx, seed=scene_seed, **replay_args)
        original_save_dir = getattr(TASK_ENV, "save_dir", None)
        try:
            TASK_ENV.save_dir = plan_save_path
            traj_data = TASK_ENV.load_tran_data(plan_episode_idx)
        finally:
            if original_save_dir is not None:
                TASK_ENV.save_dir = original_save_dir
        replay_args["left_joint_path"] = traj_data["left_joint_path"]
        replay_args["right_joint_path"] = traj_data["right_joint_path"]
        TASK_ENV.set_path_lst(replay_args)
        if progress is not None:
            progress.start_replay(output_episode_idx, trajectory_frame_count(traj_data, replay_args.get("save_freq")))

        try:
            env_info = TASK_ENV.play_once()
        except Exception as e:
            play_error = e

        try:
            task_success = play_error is None and TASK_ENV.plan_success and TASK_ENV.check_success()
        except Exception as e:
            check_error = e
            task_success = False

        if task_success:
            if not save_success:
                close_env_safely(TASK_ENV, clear_cache=((output_episode_idx + 1) % clear_cache_freq == 0))
                remove_data_cache_safely(TASK_ENV)
                finalized = True
                return "success_extra"
            save_scene_info(success_args, success_episode_idx, build_episode_info(env_info, sampler, meta, success=True))
            finalize_hdf5_episode(
                TASK_ENV,
                success_args,
                success_episode_idx,
                clear_cache=((success_episode_idx + 1) % clear_cache_freq == 0),
                require_cache=True,
            )
            finalized = True
            write_seed(success_args["save_path"], success_episode_idx, scene_seed)
            return "success"

        if not save_fail:
            print(
                f"\033[93mTask success error: skip extra failed attempt {meta.get('attempt_index')}, "
                f"seed {scene_seed}; fail target already reached\033[0m"
            )
            close_env_safely(TASK_ENV, clear_cache=((output_episode_idx + 1) % clear_cache_freq == 0))
            remove_data_cache_safely(TASK_ENV)
            finalized = True
            return "task_error_extra"
        reason = play_error or check_error or "replay_check_failed"
        print(
            f"\033[93mTask success error: save failed sample to {fail_args['save_path']} / "
            f"episode{fail_episode_idx}, seed {scene_seed}, attempt {meta.get('attempt_index')}, "
            f"reason: {reason}\033[0m"
        )
        save_traj_data_safely(TASK_ENV, fail_args["save_path"], fail_episode_idx)
        save_scene_info(fail_args, fail_episode_idx, build_episode_info(env_info, sampler, meta, success=False, failure_reason=reason))
        finalize_hdf5_episode(
            TASK_ENV,
            fail_args,
            fail_episode_idx,
            clear_cache=((fail_episode_idx + 1) % clear_cache_freq == 0),
            require_cache=False,
        )
        finalized = True
        write_seed(fail_args["save_path"], fail_episode_idx, scene_seed)
        return "task_error"
    finally:
        if not finalized:
            close_env_safely(TASK_ENV, clear_cache=((output_episode_idx + 1) % clear_cache_freq == 0))
        remove_traj_data_safely(plan_save_path, plan_episode_idx)


def collect_seed_mode(
    args,
    sampler,
    scene_seed,
    perturbation_num_success,
    perturbation_num_fail,
    max_attempts,
    start_episode=None,
    overwrite=False,
    progress=None,
):
    task = class_decorator(args["task_name"])
    scene_seed = int(scene_seed)
    success_args, fail_args = make_seed_output_args(args, scene_seed)
    plan_save_path = tempfile.mkdtemp(prefix=f"robotwin_contact_perturb_seed_{scene_seed}_")
    os.makedirs(success_args["save_path"], exist_ok=True)
    os.makedirs(fail_args["save_path"], exist_ok=True)
    try:
        if start_episode is None:
            episode_idx = next_episode_index(success_args["save_path"])
            target_episode_idx = int(perturbation_num_success)
        else:
            episode_idx = int(start_episode)
            target_episode_idx = episode_idx + int(perturbation_num_success)
        fail_episode_idx = next_episode_index(fail_args["save_path"])
        target_fail_episode_idx = int(perturbation_num_fail)

        if max_attempts is None:
            max_attempts = max(1, int(perturbation_num_success) + int(perturbation_num_fail)) * 20

        attempt_idx = 0
        success_num = 0
        planning_fail_num = 0
        task_fail_num = 0
        target_collect_num = int(perturbation_num_success) + int(perturbation_num_fail)
        if start_episode is None:
            existing_success_num = min(episode_idx, int(perturbation_num_success))
        else:
            existing_success_num = min(max(episode_idx - int(start_episode), 0), int(perturbation_num_success))
        existing_fail_num = min(fail_episode_idx, int(perturbation_num_fail))
        if progress is not None:
            progress.start_task(
                args["task_name"],
                seed=scene_seed,
                target=target_collect_num,
                target_success=int(perturbation_num_success),
                target_task_failed=int(perturbation_num_fail),
            )
            progress.update_current(
                seed=scene_seed,
                collected=existing_success_num + existing_fail_num,
                done_success=existing_success_num,
                done_task_failed=existing_fail_num,
                target=target_collect_num,
                target_success=int(perturbation_num_success),
                target_task_failed=int(perturbation_num_fail),
            )
        print("\033[93m[Start Contact-Perturbed Seed Collection]\033[0m")
        print(
            f"Fixed seed: {scene_seed}, Start episode: {episode_idx}, "
            f"Success target: {perturbation_num_success}, Fail target: {perturbation_num_fail}, "
            f"Max attempts: {max_attempts}"
        )

        while (episode_idx < target_episode_idx or fail_episode_idx < target_fail_episode_idx) and attempt_idx < max_attempts:
            meta = {
                "scene_seed": scene_seed,
                "source_mode": "seed",
                "source_task": args["task_name"],
                "source_task_config": args["source_task_config"],
                "source_episode": None,
                "perturbation_index_for_source": attempt_idx,
                "attempt_index": attempt_idx,
            }
            print(
                f"\033[34mTask: {args['task_name']}, episode: {episode_idx}, seed: {scene_seed}, "
                f"perturbation: {attempt_idx}, "
                f"success: {existing_success_num + success_num}/{perturbation_num_success}, "
                f"fail: {existing_fail_num + task_fail_num}/{perturbation_num_fail}\033[0m"
            )
            try:
                status = collect_one_seed_perturbed_episode(
                    task,
                    success_args,
                    fail_args,
                    episode_idx,
                    fail_episode_idx,
                    attempt_idx,
                    plan_save_path,
                    scene_seed,
                    sampler,
                    meta,
                    overwrite=overwrite,
                    save_success=(episode_idx < target_episode_idx),
                    save_fail=(fail_episode_idx < target_fail_episode_idx),
                    progress=progress,
                )
                if status == "success":
                    print(f"collect contact-perturbed data episode {episode_idx} success! (seed = {scene_seed})")
                    episode_idx += 1
                    success_num += 1
                elif status == "success_extra":
                    print(f"\033[93mSkip extra successful attempt {attempt_idx}: success target already reached\033[0m")
                elif status == "planning_error":
                    planning_fail_num += 1
                    print(f"\033[93mSkip attempt {attempt_idx}: planning failed\033[0m")
                elif status == "task_error":
                    task_fail_num += 1
                    fail_episode_idx += 1
                    print(
                        f"\033[93mSave failed task attempt {attempt_idx} to fail dataset "
                        f"episode {fail_episode_idx - 1}\033[0m"
                    )
                else:
                    print(f"\033[93mSkip extra failed task attempt {attempt_idx}: fail target already reached\033[0m")
                if progress is not None:
                    progress.update_current(
                        seed=scene_seed,
                        collected=existing_success_num + existing_fail_num + success_num + task_fail_num,
                        done_success=existing_success_num + success_num,
                        done_task_failed=existing_fail_num + task_fail_num,
                        target=target_collect_num,
                        target_success=int(perturbation_num_success),
                        target_task_failed=int(perturbation_num_fail),
                    )
            except Exception as e:
                print(" -------------")
                print(f"collect contact-perturbed data attempt {attempt_idx} fail! (seed = {scene_seed})")
                print("Error: ", e)
                print(traceback.format_exc())
                print(" -------------")
                close_env_safely(task)
                time.sleep(1)
            attempt_idx += 1

        print(
            f"\033[93mCollection summary: attempts={attempt_idx}, planning_failed={planning_fail_num}, "
            f"task_failed_saved={task_fail_num}, success_saved={success_num}.\033[0m"
        )
        if success_num:
            generate_episode_instructions(success_args)
        if task_fail_num:
            generate_episode_instructions(fail_args)
        if progress is not None:
            progress.finish_task()
    finally:
        shutil.rmtree(plan_save_path, ignore_errors=True)


def find_dataset_task_dir(dataset_root, task_name, source_task_config=None):
    task_root = Path(dataset_root) / task_name
    if source_task_config:
        expected = task_root / source_task_config
        if (expected / "seed.txt").is_file():
            return expected
        raise SystemExit(f"Cannot find seed.txt for task {task_name} under {expected}")

    candidates = sorted(task_root.glob("*/seed.txt"))
    if len(candidates) == 1:
        return candidates[0].parent
    if candidates:
        candidate_text = ", ".join(str(path.parent) for path in candidates[:10])
        raise SystemExit(f"Cannot choose dataset source config for {task_name}. Use --source-task-config. Candidates: {candidate_text}")
    raise SystemExit(f"Cannot find seed.txt for task {task_name} under {task_root}")


def discover_dataset_tasks(dataset_root, source_task_config=None):
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    task_names = [
        task_dir.name
        for task_dir in sorted(dataset_root.iterdir())
        if task_dir.is_dir()
        and (
            (task_dir / source_task_config / "seed.txt").is_file()
            if source_task_config
            else len(list(task_dir.glob("*/seed.txt"))) == 1
        )
    ]
    if not task_names:
        if source_task_config:
            raise SystemExit(f"Cannot find any task with {source_task_config}/seed.txt under {dataset_root}")
        raise SystemExit(f"Cannot find any task with a unique */seed.txt under {dataset_root}")
    return task_names


def collect_dataset_mode(
    dataset_root,
    task_names,
    task_config,
    source_task_config,
    sampler,
    sample_start,
    sample_end,
    perturbation_num_success,
    perturbation_num_fail,
    max_attempts,
    start_episode=None,
    overwrite=False,
    progress=None,
):
    perturbation_num = int(perturbation_num_success) + int(perturbation_num_fail)
    if max_attempts is None:
        max_attempts = int(perturbation_num) * 20

    for task_name in task_names:
        sample_num = sample_end - sample_start + 1
        task_success_target_num = int(perturbation_num_success) * sample_num
        task_fail_target_num = int(perturbation_num_fail) * sample_num
        source_dir = find_dataset_task_dir(dataset_root, task_name, source_task_config)
        resolved_source_task_config = source_dir.name
        seeds = read_seed_list(source_dir)
        if sample_start < 0 or sample_end < sample_start:
            raise SystemExit("--sample-start must be >= 0 and --sample-end must be >= sample-start")
        if sample_end >= len(seeds):
            raise SystemExit(f"{source_dir}/seed.txt has {len(seeds)} seeds, cannot use sample-end {sample_end}")

        args = build_args(
            task_name,
            task_config,
            output_save_root=ROBOTWIN_ROOT / "data" / task_config,
        )
        task = class_decorator(task_name)
        plan_save_path = tempfile.mkdtemp(prefix=f"robotwin_contact_perturb_dataset_{task_name}_")
        success_num = 0
        planning_fail_num = 0
        task_fail_num = 0
        task_targets = {}
        task_target_num = 0
        task_existing_success_num = 0
        task_existing_fail_num = 0
        success_target_per_source = int(perturbation_num_success)
        fail_target_per_source = int(perturbation_num_fail)
        for source_episode in range(sample_start, sample_end + 1):
            source_seed = seeds[source_episode]
            success_args, fail_args = make_seed_output_args(args, source_seed)
            os.makedirs(success_args["save_path"], exist_ok=True)
            os.makedirs(fail_args["save_path"], exist_ok=True)
            collected_success_keys = set() if overwrite else read_collected_source_keys(success_args["save_path"])
            collected_fail_keys = set() if overwrite else read_collected_source_keys(fail_args["save_path"])
            existing_success_indices = {
                perturb_idx
                for collected_task, collected_episode, perturb_idx in collected_success_keys
                if collected_task == task_name and collected_episode == source_episode
            }
            existing_fail_indices = {
                perturb_idx
                for collected_task, collected_episode, perturb_idx in collected_fail_keys
                if collected_task == task_name and collected_episode == source_episode
            }
            existing_success_indices = {idx for idx in existing_success_indices if idx >= 0}
            existing_fail_indices = {idx for idx in existing_fail_indices if idx >= 0}
            existing_success_count = len(existing_success_indices)
            existing_fail_count = len(existing_fail_indices)
            task_existing_success_num += min(existing_success_count, success_target_per_source)
            task_existing_fail_num += min(existing_fail_count, fail_target_per_source)
            existing_indices = existing_success_indices | existing_fail_indices
            next_perturb_idx = max(existing_indices) + 1 if existing_indices else 0
            task_targets[source_episode] = {
                "existing_success": existing_success_count,
                "existing_fail": existing_fail_count,
                "next_perturb_idx": next_perturb_idx,
            }
            task_target_num += int(perturbation_num)
        if progress is not None:
            progress.start_task(
                task_name,
                seed="-",
                target=task_target_num,
                target_success=task_success_target_num,
                target_task_failed=task_fail_target_num,
            )
            progress.update_current(
                seed="-",
                collected=task_existing_success_num + task_existing_fail_num,
                done_success=task_existing_success_num,
                done_task_failed=task_existing_fail_num,
                target=task_target_num,
                target_success=task_success_target_num,
                target_task_failed=task_fail_target_num,
            )

        print("\033[93m[Start Contact-Perturbed Dataset Collection]\033[0m")
        print(
            f"Task: {task_name}, Source: {source_dir}, Output: {args['save_path']}, "
            f"Samples: {sample_start}-{sample_end}, Success target: {perturbation_num_success}, "
            f"Fail target: {perturbation_num_fail}, Max attempts: {max_attempts}"
        )

        for source_episode in range(sample_start, sample_end + 1):
            scene_seed = seeds[source_episode]
            success_args, fail_args = make_seed_output_args(args, scene_seed)
            os.makedirs(success_args["save_path"], exist_ok=True)
            os.makedirs(fail_args["save_path"], exist_ok=True)
            episode_idx = next_episode_index(success_args["save_path"]) if start_episode is None else int(start_episode)
            fail_episode_idx = next_episode_index(fail_args["save_path"])
            source_target = task_targets[source_episode]
            source_existing_success_num = int(source_target["existing_success"])
            source_existing_fail_num = int(source_target["existing_fail"])
            next_perturb_idx = int(source_target["next_perturb_idx"])
            if (
                source_existing_success_num >= success_target_per_source
                and source_existing_fail_num >= fail_target_per_source
            ):
                print(
                    f"\033[93mSkip source episode {source_episode}: "
                    f"success {source_existing_success_num}/{success_target_per_source}, "
                    f"fail {source_existing_fail_num}/{fail_target_per_source}\033[0m"
                )
                continue

            if progress is not None:
                progress.update_current(
                    seed=scene_seed,
                    target=task_target_num,
                    target_success=task_success_target_num,
                    target_task_failed=task_fail_target_num,
                )
            attempt_idx = 0
            source_success_num = 0
            source_task_fail_num = 0
            while (
                (
                    source_existing_success_num + source_success_num < success_target_per_source
                    or source_existing_fail_num + source_task_fail_num < fail_target_per_source
                )
                and attempt_idx < max_attempts
            ):
                perturb_idx = next_perturb_idx + attempt_idx
                source_key = (task_name, source_episode, perturb_idx)
                save_success = source_existing_success_num + source_success_num < success_target_per_source
                save_fail = source_existing_fail_num + source_task_fail_num < fail_target_per_source

                meta = {
                    "scene_seed": scene_seed,
                    "source_mode": "dataset",
                    "source_task": task_name,
                    "source_task_config": resolved_source_task_config,
                    "source_episode": source_episode,
                    "perturbation_index_for_source": perturb_idx,
                    "attempt_index": attempt_idx,
                }
                print(
                    f"\033[34mTask: {task_name}, episode: {episode_idx}, seed: {scene_seed}, "
                    f"perturbation: {perturb_idx} item {source_episode}, "
                    f"source success: {source_existing_success_num + source_success_num}/{success_target_per_source}, "
                    f"source fail: {source_existing_fail_num + source_task_fail_num}/{fail_target_per_source}, "
                    f"task success: {task_existing_success_num + success_num}/{task_success_target_num}, "
                    f"task fail: {task_existing_fail_num + task_fail_num}/{task_fail_target_num}\033[0m"
                )
                try:
                    status = collect_one_seed_perturbed_episode(
                        task,
                        success_args,
                        fail_args,
                        episode_idx,
                        fail_episode_idx,
                        attempt_idx,
                        plan_save_path,
                        scene_seed,
                        sampler,
                        meta,
                        overwrite=overwrite,
                        save_success=save_success,
                        save_fail=save_fail,
                        progress=progress,
                    )
                    if status == "success":
                        print(f"collect contact-perturbed data episode {episode_idx} success! (seed = {scene_seed})")
                        episode_idx += 1
                        success_num += 1
                        source_success_num += 1
                    elif status == "success_extra":
                        print(f"\033[93mSkip extra successful source key {source_key}: success target already reached\033[0m")
                    elif status == "planning_error":
                        planning_fail_num += 1
                        print(f"\033[93mSkip source key {source_key}: planning failed\033[0m")
                    elif status == "task_error":
                        task_fail_num += 1
                        source_task_fail_num += 1
                        fail_episode_idx += 1
                        print(f"\033[93mSave failed source key {source_key} to fail dataset\033[0m")
                    else:
                        print(f"\033[93mSkip extra failed source key {source_key}: fail target already reached\033[0m")
                    if progress is not None:
                        progress.update_current(
                            seed=scene_seed,
                            collected=task_existing_success_num + task_existing_fail_num + success_num + task_fail_num,
                            done_success=task_existing_success_num + success_num,
                            done_task_failed=task_existing_fail_num + task_fail_num,
                            target=task_target_num,
                            target_success=task_success_target_num,
                            target_task_failed=task_fail_target_num,
                        )
                except Exception as e:
                    print(" -------------")
                    print(f"collect contact-perturbed data source key {source_key} fail! (seed = {scene_seed})")
                    print("Error: ", e)
                    print(traceback.format_exc())
                    print(" -------------")
                    close_env_safely(task)
                    time.sleep(1)
                attempt_idx += 1

            missing_success_num = max(
                success_target_per_source - source_existing_success_num - source_success_num,
                0,
            )
            missing_fail_num = max(
                fail_target_per_source - source_existing_fail_num - source_task_fail_num,
                0,
            )
            if missing_success_num or missing_fail_num:
                print(
                    f"\033[93mSource episode {source_episode} reached max attempts {max_attempts}; "
                    f"missing success={missing_success_num}, missing fail={missing_fail_num}\033[0m"
                )
            if source_success_num:
                generate_episode_instructions(success_args)
            if source_task_fail_num:
                generate_episode_instructions(fail_args)

        print(
            f"\033[93mTask summary: planning_failed={planning_fail_num}, "
            f"task_failed={task_fail_num}, success={success_num}.\033[0m"
        )
        if progress is not None:
            progress.finish_task()
        shutil.rmtree(plan_save_path, ignore_errors=True)


def parse_range(values, name):
    if values is None:
        return [0.0, 0.0]
    if len(values) != 2:
        raise SystemExit(f"{name} requires exactly two values: MIN MAX")
    return [float(values[0]), float(values[1])]


def build_ranges(cli_args):
    return {
        "x": parse_range(cli_args.x_range, "--x-range"),
        "y": parse_range(cli_args.y_range, "--y-range"),
        "z": parse_range(cli_args.z_range, "--z-range"),
        "roll_deg": parse_range(cli_args.roll_range, "--roll-range"),
        "pitch_deg": parse_range(cli_args.pitch_range, "--pitch-range"),
        "yaw_deg": parse_range(cli_args.yaw_range, "--yaw-range"),
    }


def resolve_perturbation_targets(cli_args):
    total = cli_args.perturbation_num
    success = cli_args.perturbation_num_success
    fail = cli_args.perturbation_num_fail

    for name, value in (
        ("--perturbation-num", total),
        ("--perturbation-num-success", success),
        ("--perturbation-num-fail", fail),
    ):
        if value is not None and value < 0:
            raise SystemExit(f"{name} must be >= 0")

    if total is None:
        if success is None or fail is None:
            raise SystemExit(
                "Requires either --perturbation-num, or both "
                "--perturbation-num-success and --perturbation-num-fail"
            )
        total = success + fail
    else:
        if success is None and fail is None:
            success = total
            fail = 0
        elif success is None:
            success = total - fail
        elif fail is None:
            fail = total - success
        if success < 0 or fail < 0:
            raise SystemExit("--perturbation-num must be >= provided success/fail target")
        if success + fail != total:
            raise SystemExit(
                "--perturbation-num-success + --perturbation-num-fail must equal --perturbation-num"
            )

    if total <= 0:
        raise SystemExit("At least one perturbation must be requested")
    return int(success), int(fail)


def main(cli_args):
    if cli_args.max_attempts is not None and cli_args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be positive")

    perturb_seed = cli_args.perturb_seed
    if perturb_seed is None:
        perturb_seed = cli_args.seed if cli_args.seed is not None else 0
    sampler = ContactPointPerturbationSampler(
        build_ranges(cli_args),
        seed=perturb_seed,
        fixed_episode_perturbation=not cli_args.dynamic_call_perturbation,
    )

    progress = None
    install_contact_point_perturbation_patch(sampler)
    try:
        if cli_args.mode == "seed":
            if cli_args.task_name is None or cli_args.task_config is None or cli_args.seed is None:
                raise SystemExit("--mode seed requires --task-name, --task-config, and --seed")
            perturbation_num_success, perturbation_num_fail = resolve_perturbation_targets(cli_args)
            args = build_args(
                cli_args.task_name,
                cli_args.task_config,
            )
            progress = OverallCollectionProgress(total_tasks=1)
            with progress.redirect_output():
                collect_seed_mode(
                    args,
                    sampler,
                    scene_seed=cli_args.seed,
                    perturbation_num_success=perturbation_num_success,
                    perturbation_num_fail=perturbation_num_fail,
                    max_attempts=cli_args.max_attempts,
                    start_episode=cli_args.start_episode,
                    overwrite=cli_args.overwrite,
                    progress=progress,
                )
        else:
            if cli_args.task_config is None:
                raise SystemExit("--mode dataset requires --task-config")
            task_names = cli_args.tasks or discover_dataset_tasks(cli_args.dataset_root, cli_args.source_task_config)
            perturbation_num_success, perturbation_num_fail = resolve_perturbation_targets(cli_args)
            progress = OverallCollectionProgress(total_tasks=len(task_names))
            with progress.redirect_output():
                collect_dataset_mode(
                    dataset_root=cli_args.dataset_root,
                    task_names=task_names,
                    task_config=cli_args.task_config,
                    source_task_config=cli_args.source_task_config,
                    sampler=sampler,
                    sample_start=cli_args.sample_start,
                    sample_end=cli_args.sample_end,
                    perturbation_num_success=perturbation_num_success,
                    perturbation_num_fail=perturbation_num_fail,
                    max_attempts=cli_args.max_attempts,
                    start_episode=cli_args.start_episode,
                    overwrite=cli_args.overwrite,
                    progress=progress,
                )
    finally:
        if progress is not None:
            progress.close()
        uninstall_contact_point_perturbation_patch()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "dataset"], default="seed")
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--task-config", type=str, default=None)
    parser.add_argument("--source-task-config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--perturb-seed", type=int, default=None)
    parser.add_argument("--perturbation-num", type=int, default=None)
    parser.add_argument("--perturbation-num-success", type=int, default=None)
    parser.add_argument("--perturbation-num-fail", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=0)
    parser.add_argument("--start-episode", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dynamic-call-perturbation", action="store_true")
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.0, 0.0], metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", nargs=2, type=float, default=[0.0, 0.0], metavar=("MIN", "MAX"))
    parser.add_argument("--z-range", nargs=2, type=float, default=[0.0, 0.0], metavar=("MIN", "MAX"))
    parser.add_argument("--roll-range", nargs=2, type=float, default=[0.0, 0.0], metavar=("MIN_DEG", "MAX_DEG"))
    parser.add_argument("--pitch-range", nargs=2, type=float, default=[0.0, 0.0], metavar=("MIN_DEG", "MAX_DEG"))
    parser.add_argument("--yaw-range", nargs=2, type=float, default=[0.0, 0.0], metavar=("MIN_DEG", "MAX_DEG"))
    args = parser.parse_args()

    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main(args)
