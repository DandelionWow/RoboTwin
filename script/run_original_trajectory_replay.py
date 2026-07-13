#!/usr/bin/env python3
"""Replay original RoboTwin trajectories with multiple GPU workers."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/data/wangbowen/PHD_Research/02_Action_Generalization/Action_Hallucination_Verification/robotwin_consistency/database/datasets/robotwin_480_640")
ACTION_FIELDS = [
    "left_arm",
    "left_gripper",
    "right_arm",
    "right_gripper",
    "vector",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用原始 seed 和 _traj_data 并行执行严格轨迹 replay."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-variant", default="demo_large_d435_replay")
    parser.add_argument("--output-name", required=True, help="输出目录名, 同时也是任务配置名.")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--tasks", nargs="*", help="不传时自动发现数据集中的全部任务.")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只准备 seed、轨迹和配置, 不启动仿真.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_config(episodes: int) -> dict:
    return {
        "render_freq": 0,
        "episode_num": episodes,
        "use_seed": True,
        "save_freq": 15,
        "embodiment": ["aloha-agilex"],
        "language_num": 100,
        "domain_randomization": {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1,
            "random_head_camera_dis": 0,
            "random_table_height": 0,
            "random_light": False,
            "crazy_random_light_rate": 0,
        },
        "camera": {
            "head_camera_type": "Large_D435",
            "wrist_camera_type": "Large_D435",
            "seed_compatible_static_camera_num": 2,
            "collect_head_camera": True,
            "collect_wrist_camera": True,
        },
        "data_type": {
            "rgb": True,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "observer": False,
            "endpose": True,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        },
        "pcd_down_sample_num": 1024,
        "pcd_crop": True,
        "save_path": "./data",
        "clear_cache_freq": 5,
        "collect_data": True,
        "eval_video_log": True,
        "replay_scene_info": True,
    }


def write_or_validate_config(output_name: str, episodes: int) -> Path:
    config_path = PROJECT_ROOT / "task_config" / f"{output_name}.yml"
    expected = replay_config(episodes)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            actual = yaml.safe_load(file)
        if actual != expected:
            raise ValueError(f"已有配置与预期 replay 配置不同: {config_path}")
        return config_path

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(expected, file, allow_unicode=True, sort_keys=False)
    return config_path


def discover_tasks(dataset_root: Path, dataset_variant: str) -> list[str]:
    tasks = [
        path.parent.name
        for path in dataset_root.glob(f"*/{dataset_variant}")
        if path.is_dir()
    ]
    if not tasks:
        raise FileNotFoundError(
            f"没有在 {dataset_root} 中找到数据集变体 {dataset_variant}"
        )
    return sorted(tasks)


def copy_or_validate(source: Path, target: Path) -> None:
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError(f"目标文件已存在但内容不同: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def replay_hdf5_is_complete(source: Path, target: Path) -> bool:
    if not target.is_file():
        return False
    try:
        with h5py.File(source, "r") as source_file, h5py.File(target, "r") as target_file:
            for field in ACTION_FIELDS:
                expected = source_file[f"joint_action/{field}"][()]
                actual = target_file[f"joint_action/{field}"][()]
                if expected.shape != actual.shape or not np.array_equal(expected, actual):
                    return False
            frame_count = len(target_file["joint_action/vector"])
            if len(target_file["observation/head_camera/rgb"]) != frame_count:
                return False
    except (KeyError, OSError):
        return False
    return True


def prepare_task(
    task: str,
    dataset_root: Path,
    dataset_variant: str,
    output_name: str,
    episodes: int,
) -> Path:
    source = dataset_root / task / dataset_variant
    if not source.is_dir():
        raise FileNotFoundError(f"原始任务目录不存在: {source}")

    seeds = (source / "seed.txt").read_text(encoding="utf-8").split()
    if len(seeds) < episodes:
        raise ValueError(f"{source / 'seed.txt'} 只有 {len(seeds)} 个 seed")

    output = PROJECT_ROOT / "data" / task / output_name
    output.mkdir(parents=True, exist_ok=True)
    seed_path = output / "seed.txt"
    expected_seed_text = " ".join(seeds[:episodes]) + " "
    if seed_path.exists():
        existing_seeds = seed_path.read_text(encoding="utf-8").split()
        if existing_seeds[:episodes] != seeds[:episodes]:
            raise ValueError(f"已有 seed 文件前 {episodes} 项内容不同: {seed_path}")
    else:
        seed_path.write_text(expected_seed_text, encoding="utf-8")

    for episode in range(episodes):
        copy_or_validate(
            source / "_traj_data" / f"episode{episode}.pkl",
            output / "_traj_data" / f"episode{episode}.pkl",
        )
        hdf5_path = output / "data" / f"episode{episode}.hdf5"
        if hdf5_path.exists() and not replay_hdf5_is_complete(
            source / "data" / f"episode{episode}.hdf5", hdf5_path
        ):
            print(f"删除不完整或 action 不一致的输出: {hdf5_path}")
            hdf5_path.unlink()
            video_path = output / "video" / f"episode{episode}.mp4"
            if video_path.exists():
                video_path.unlink()
    copy_or_validate(
        source / "scene_info.json",
        output / "_replay_source" / "scene_info.json",
    )
    return output


def output_complete(output: Path, episodes: int) -> bool:
    return all(
        (output / "data" / f"episode{episode}.hdf5").is_file()
        for episode in range(episodes)
    )


def run_task(task: str, output_name: str, gpu: int, episodes: int) -> str:
    output = PROJECT_ROOT / "data" / task / output_name
    if output_complete(output, episodes):
        return f"{task}: 已完成, 跳过"

    log_path = PROJECT_ROOT / "data" / "_logs" / output_name / f"{task}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "bash",
        "collect_data.sh",
        task,
        output_name,
        str(gpu),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    # collect_data.sh ends with rm, so its exit code may hide a Python failure.
    if result.returncode != 0 or not output_complete(output, episodes):
        raise RuntimeError(f"{task} replay 失败, 请检查 {log_path}")
    return f"{task}: 完成, GPU {gpu}"


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes 必须大于 0")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu 必须大于 0")

    dataset_root = args.dataset_root.resolve()
    tasks = args.tasks or discover_tasks(dataset_root, args.dataset_variant)
    write_or_validate_config(args.output_name, args.episodes)

    outputs = {}
    for task in tasks:
        outputs[task] = prepare_task(
            task,
            dataset_root,
            args.dataset_variant,
            args.output_name,
            args.episodes,
        )
    print(f"已准备 {len(outputs)} 个任务: data/*/{args.output_name}")

    if args.prepare_only:
        return

    gpu_slots = [
        gpu for _ in range(args.workers_per_gpu) for gpu in args.gpus
    ]
    failures = []
    with ThreadPoolExecutor(max_workers=len(gpu_slots)) as executor:
        futures = {}
        for index, task in enumerate(tasks):
            gpu = gpu_slots[index % len(gpu_slots)]
            future = executor.submit(
                run_task, task, args.output_name, gpu, args.episodes
            )
            futures[future] = task
        for future in as_completed(futures):
            task = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                failures.append(task)
                print(f"{task}: 失败: {error}", flush=True)

    if failures:
        raise SystemExit(f"失败任务 ({len(failures)}): {', '.join(sorted(failures))}")


if __name__ == "__main__":
    main()
