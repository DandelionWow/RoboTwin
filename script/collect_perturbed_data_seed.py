import json
import os
import sys
import time
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Disable GPU usage for this script

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROBOTWIN_ROOT)
sys.path.append(str(ROBOTWIN_ROOT))

import importlib
import yaml

CONFIGS_PATH = str(ROBOTWIN_ROOT / "task_config")


DEFAULT_WAYPOINT_CACHE_ROOT = (
    "/data/wangbowen/PHD_Research/02_Action_Generalization/Action_Hallucination_Verification/"
    "robotwin_consistency/database/waypoint_selection_cache"
)


class TaskSuccessCheckError(Exception):
    pass


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        return getattr(envs_module, task_name)()
    except Exception:
        raise SystemExit("No such task")


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


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


def perturbation_record_key(record):
    return (
        str(record.get("save_id")),
        int(record.get("item_index", -1)),
        int(record.get("point_id", -1)),
    )


def read_collected_perturbation_keys(save_path):
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
        save_id = info.get("perturbation_save_id")
        item_index = info.get("perturbation_item_index")
        point_id = info.get("perturbation_point_id")
        if save_id is None or item_index is None or point_id is None:
            continue
        keys.add((str(save_id), int(item_index), int(point_id)))
    return keys


def finalize_hdf5_episode(TASK_ENV, args, episode_idx, clear_cache=False):
    original_save_dir = getattr(TASK_ENV, "save_dir", None)
    original_ep_num = getattr(TASK_ENV, "ep_num", None)
    closed = False
    try:
        TASK_ENV.save_dir = args["save_path"]
        TASK_ENV.ep_num = episode_idx
        TASK_ENV.close_env(clear_cache=clear_cache)
        closed = True
        TASK_ENV.merge_pkl_to_hdf5_video()
        remove_data_cache_safely(TASK_ENV)
    finally:
        if not closed:
            close_env_safely(TASK_ENV, clear_cache=clear_cache)
        if original_save_dir is not None:
            TASK_ENV.save_dir = original_save_dir
        if original_ep_num is not None:
            TASK_ENV.ep_num = original_ep_num


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


def generate_episode_instructions(args):
    from description.utils.generate_episode_instructions import (
        extract_episodes_from_scene_info,
        generate_episode_descriptions,
        load_scene_info,
        save_episode_descriptions,
    )

    config_name = args.get("source_task_config", args["task_config"])
    config_path = ROBOTWIN_ROOT / "task_config" / f"{config_name}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    setting = args["task_config"]
    scene_info = load_scene_info(args["task_name"], setting, config_args["save_path"])
    episodes = extract_episodes_from_scene_info(scene_info)
    descriptions = generate_episode_descriptions(args["task_name"], episodes, args["language_num"])
    save_episode_descriptions(args["task_name"], setting, descriptions, config_args["save_path"])


def load_perturbation_records(cache_root, task_config, task_name, seed, limit=None):
    perturb_root = Path(cache_root) / task_name / task_config / f"seed_{seed}" / "perturbation_saves"
    if not perturb_root.is_dir():
        print(f"\033[93mWarning: perturbation_saves not found: {perturb_root}\033[0m")
        return []

    records = []
    for save_dir in sorted(path for path in perturb_root.iterdir() if path.is_dir()):
        poses_path = save_dir / "poses.json"
        if not poses_path.is_file():
            print(f"\033[93mWarning: skip {save_dir.name}, poses.json not found\033[0m")
            continue
        try:
            data = json.loads(poses_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"\033[93mWarning: skip {poses_path}: {e}\033[0m")
            continue
        for item_index, item in enumerate(data.get("items", [])):
            pre_pose = item.get("perturbed_pre_grasp_pose_world")
            grasp_pose = item.get("perturbed_grasp_pose_world")
            if not pre_pose or not grasp_pose:
                print(f"\033[93mWarning: skip {save_dir.name} item {item_index}, missing grasp poses\033[0m")
                continue
            records.append({
                "save_id": save_dir.name,
                "item_index": item_index,
                "point_id": item.get("point_id"),
                "pre_grasp_dis": item.get("pre_grasp_dis", data.get("pre_grasp_dis")),
                "grasp_dis": item.get("grasp_dis", data.get("grasp_dis")),
                "pre_grasp_pose_world": pre_pose,
                "grasp_pose_world": grasp_pose,
                "tcp_pose_world": item.get("perturbed_tcp_pose_world"),
                "source_dir": str(save_dir),
            })
            if limit is not None and len(records) >= limit:
                return records
    return records


def add_perturbation_info(info, seed, record):
    if not isinstance(info, dict):
        info = {}
    info = deepcopy(info)
    info["seed"] = seed
    info["perturbation_save_id"] = record["save_id"]
    info["perturbation_item_index"] = record["item_index"]
    info["perturbation_point_id"] = record["point_id"]
    info["perturbation_source_dir"] = record["source_dir"]
    if record.get("pre_grasp_dis") is not None:
        info["pre_grasp_dis"] = record["pre_grasp_dis"]
    if record.get("grasp_dis") is not None:
        info["grasp_dis"] = record["grasp_dis"]
    info["pre_grasp_pose_world"] = record.get("pre_grasp_pose_world")
    info["grasp_pose_world"] = record.get("grasp_pose_world")
    info["tcp_pose_world"] = record.get("tcp_pose_world")
    return info


def collect_pre_motion_for_record(TASK_ENV, args, episode_idx, seed, record, render_freq, traj_args, traj_episode_idx):
    args["need_plan"] = True
    args["save_data"] = False
    args["render_freq"] = render_freq
    args["perturbed_grasp_record"] = record
    args.pop("left_joint_path", None)
    args.pop("right_joint_path", None)

    original_save_dir = getattr(TASK_ENV, "save_dir", None)
    try:
        TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
        TASK_ENV.play_once()
        if not TASK_ENV.plan_success:
            print(f"\033[93mPlanning error: episode {episode_idx}, perturbation {record['save_id']} item {record['item_index']}\033[0m")
            return "planning_error"
        try:
            task_success = TASK_ENV.check_success()
        except Exception as e:
            raise TaskSuccessCheckError(f"Task success check crashed during planning: {e}") from e
        if not task_success:
            print(
                f"\033[93mTask success error during planning: episode {episode_idx}, "
                f"perturbation {record['save_id']} item {record['item_index']}; "
                "keep trajectory for failed-sample replay\033[0m"
            )
        TASK_ENV.save_dir = traj_args["save_path"]
        TASK_ENV.save_traj_data(traj_episode_idx)
        return "success"
    finally:
        if original_save_dir is not None:
            TASK_ENV.save_dir = original_save_dir
        close_env_safely(TASK_ENV, render_freq=render_freq)


def collect_hdf5_for_record(
    TASK_ENV,
    args,
    episode_idx,
    seed,
    record,
    success_args,
    success_episode_idx,
    fail_args,
    fail_episode_idx,
    clear_cache=False,
):
    replay_args = args.copy()
    replay_args["need_plan"] = False
    replay_args["save_data"] = True
    replay_args["render_freq"] = 0
    replay_args["perturbed_grasp_record"] = record
    replay_args["save_path"] = success_args["save_path"]

    finalized = False
    try:
        TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed, **replay_args)
        traj_data = TASK_ENV.load_tran_data(success_episode_idx)
        replay_args["left_joint_path"] = traj_data["left_joint_path"]
        replay_args["right_joint_path"] = traj_data["right_joint_path"]
        TASK_ENV.set_path_lst(replay_args)

        info = TASK_ENV.play_once()
        if not TASK_ENV.plan_success:
            print(f"\033[93mReplay planning error: episode {episode_idx}, perturbation {record['save_id']} item {record['item_index']}\033[0m")
            return "planning_error"

        try:
            task_success = TASK_ENV.check_success()
        except Exception as e:
            raise TaskSuccessCheckError(f"Task success check crashed during replay: {e}") from e
        if task_success:
            target_args = success_args
            output_episode_idx = success_episode_idx
        else:
            target_args = fail_args
            output_episode_idx = fail_episode_idx
            TASK_ENV.save_dir = fail_args["save_path"]
            TASK_ENV.save_traj_data(fail_episode_idx)

        save_scene_info(target_args, output_episode_idx, add_perturbation_info(info, seed, record))
        finalize_hdf5_episode(TASK_ENV, target_args, output_episode_idx, clear_cache=clear_cache)
        finalized = True
        write_seed(target_args["save_path"], output_episode_idx, seed)
        if not task_success:
            print(f"\033[93mTask success error: save failed sample to {target_args['save_path']} / episode{output_episode_idx}\033[0m")
            return "task_error"
        return "success"
    finally:
        if not finalized:
            close_env_safely(TASK_ENV, clear_cache=clear_cache)


def collect_perturbed_data(TASK_ENV, args, seed, records, start_episode=None, overwrite=False):
    success_args = args.copy()
    success_args["task_config"] = f"{args['task_config']}_success"
    success_args["save_path"] = f"{args['save_path']}_success"
    fail_args = args.copy()
    fail_args["task_config"] = f"{args['task_config']}_fail"
    fail_args["save_path"] = f"{args['save_path']}_fail"
    os.makedirs(success_args["save_path"], exist_ok=True)
    os.makedirs(fail_args["save_path"], exist_ok=True)

    if start_episode is None:
        start_episode = next_episode_index(success_args["save_path"])
    episode_idx = start_episode
    success_episode_idx = start_episode
    fail_episode_idx = next_episode_index(fail_args["save_path"])
    plan_render_freq = args.get("render_freq", 0)
    clear_cache_freq = args.get("clear_cache_freq", 1) or 1
    if overwrite:
        collected_keys = set()
    else:
        collected_keys = (
            read_collected_perturbation_keys(success_args["save_path"])
            | read_collected_perturbation_keys(fail_args["save_path"])
        )
    total_num = 0
    planning_fail_num = 0
    task_fail_num = 0
    success_num = 0

    print("\033[93m[Start Perturbed Data Collection From Fixed Seed]\033[0m")
    print(f"Fixed seed: {seed}, Start episode: {episode_idx}, Records: {len(records)}")

    for record in records:
        record_key = perturbation_record_key(record)
        if record_key in collected_keys:
            print(
                f"\033[93mSkip collected perturbation {record['save_id']} "
                f"item {record['item_index']} point {record.get('point_id')}\033[0m"
            )
            continue

        total_num += 1
        ensure_episode_writable(success_args["save_path"], success_episode_idx, overwrite)
        ensure_episode_writable(fail_args["save_path"], fail_episode_idx, overwrite)
        print(
            f"\033[34mTask: {args['task_name']}, episode: {episode_idx}, seed: {seed}, "
            f"perturbation: {record['save_id']} item {record['item_index']}\033[0m"
        )
        try:
            pre_motion_status = collect_pre_motion_for_record(
                TASK_ENV,
                args,
                episode_idx,
                seed,
                record,
                plan_render_freq,
                success_args,
                success_episode_idx,
            )
            if pre_motion_status == "planning_error":
                planning_fail_num += 1
                print(f"\033[93mSkip episode {episode_idx}: planning failed\033[0m")
                continue
            if pre_motion_status == "task_error":
                task_fail_num += 1
                print(f"\033[93mSkip episode {episode_idx}: task success check failed during planning\033[0m")
                continue

            replay_status = collect_hdf5_for_record(
                TASK_ENV,
                args,
                episode_idx,
                seed,
                record,
                success_args,
                success_episode_idx,
                fail_args,
                fail_episode_idx,
                clear_cache=((episode_idx + 1) % clear_cache_freq == 0),
            )
            if replay_status == "planning_error":
                planning_fail_num += 1
                print(f"\033[93mSkip episode {episode_idx}: replay planning failed\033[0m")
                continue
            if replay_status == "task_error":
                task_fail_num += 1
                fail_episode_idx += 1
                continue

            print(f"collect perturbed data episode {episode_idx} success! (fixed seed = {seed})")
            collected_keys.add(record_key)
            success_episode_idx += 1
            episode_idx += 1
            success_num += 1
        except TaskSuccessCheckError:
            close_env_safely(TASK_ENV, render_freq=plan_render_freq)
            raise
        except Exception as e:
            print(" -------------")
            print(f"collect perturbed data episode {episode_idx} fail! (fixed seed = {seed})")
            print("Error: ", e)
            print(traceback.format_exc())
            print(" -------------")
            close_env_safely(TASK_ENV, render_freq=plan_render_freq)
            time.sleep(1)

    print(
        f"\033[93mCollection summary: total={total_num}, planning_failed={planning_fail_num}, "
        f"task_failed={task_fail_num}, success={success_num}.\033[0m"
    )
    if success_num:
        generate_episode_instructions(success_args)


def build_args(task_name, task_config, seed, episode_num=None):
    config_path = ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if episode_num is not None:
        args["episode_num"] = int(episode_num)

    args["task_name"] = task_name
    original_task_config = task_config
    args["task_config"] = os.path.join(original_task_config, f"seed_{seed}")

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
    args["save_path"] = os.path.join(args["save_path"], task_name, args["task_config"])
    args["source_task_config"] = original_task_config
    return args


def main(task_name, task_config, seed, waypoint_cache_root, start_episode=None, overwrite=False, limit=None, episode_num=None):
    task = class_decorator(task_name)
    records = load_perturbation_records(waypoint_cache_root, task_config, task_name, seed, limit=limit)
    if episode_num is not None:
        records = records[:episode_num]
    if not records:
        print("\033[93mNo perturbation records to collect.\033[0m")
        return

    args = build_args(task_name, task_config, seed, episode_num=episode_num)
    collect_perturbed_data(task, args, seed, records, start_episode=start_episode, overwrite=overwrite)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task-name", required=True, type=str)
    parser.add_argument("--task-config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--waypoint-cache-root", default=DEFAULT_WAYPOINT_CACHE_ROOT)
    parser.add_argument("--start-episode", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--episode-num", type=int, default=None)
    args = parser.parse_args()

    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main(
        task_name=args.task_name,
        task_config=args.task_config,
        seed=args.seed,
        waypoint_cache_root=args.waypoint_cache_root,
        start_episode=args.start_episode,
        overwrite=args.overwrite,
        limit=args.limit,
        episode_num=args.episode_num,
    )
