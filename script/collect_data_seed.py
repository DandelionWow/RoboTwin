import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    """根据任务名称加载并实例化任务环境类。

    Args:
        task_name: `envs` 目录下的任务模块名和类名，例如
            `place_object_basket`。

    Returns:
        已实例化的任务环境对象。

    Raises:
        SystemExit: 任务模块或任务类无法加载时抛出。
    """
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    """读取单个机器人 embodiment 的配置文件。

    Args:
        robot_file: 机器人 embodiment 资源目录路径。

    Returns:
        从 `<robot_file>/config.yml` 解析得到的配置字典。
    """
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def read_seed_list(save_path):
    """读取正常成功 episode 的 seed 列表。

    Args:
        save_path: 数据集保存目录，目录下可能包含 `seed.txt`。

    Returns:
        按 episode 顺序排列的整数 seed 列表；如果 `seed.txt` 不存在，
        返回空列表。
    """
    seed_file_path = os.path.join(save_path, "seed.txt")
    if not os.path.exists(seed_file_path):
        return []

    with open(seed_file_path, "r", encoding="utf-8") as file:
        return [int(seed) for seed in file.read().split()]


def write_seed_list(save_path, seed_list):
    """将成功 episode 的 seed 列表写入 `seed.txt`。

    Args:
        save_path: 数据集保存目录。
        seed_list: 按正常 episode 下标排序的 seed 列表。

    Returns:
        None。该函数会更新 `<save_path>/seed.txt`。
    """
    with open(os.path.join(save_path, "seed.txt"), "w", encoding="utf-8") as file:
        file.write(" ".join(str(seed) for seed in seed_list))
        if seed_list:
            file.write(" ")


def parse_seed_values(seed_values):
    """从命令行参数中解析 seed 值。

    Args:
        seed_values: seed 字符串序列。每个字符串可以是单个 seed，
            也可以包含逗号或空格分隔的多个 seed。

    Returns:
        校验后的整数 seed 列表。

    Raises:
        SystemExit: 任意 seed 超出 `[0, 2**32 - 1]` 时抛出。
    """
    seeds = []
    for value in seed_values:
        for seed in str(value).replace(",", " ").split():
            seed = int(seed)
            if seed < 0 or seed > 2**32 - 1:
                raise SystemExit(f"Seed must be in [0, {2**32 - 1}], got {seed}")
            seeds.append(seed)
    return seeds


def check_seed_slot(seed_list, episode_idx, seed, overwrite):
    """校验某个 episode 下标是否可以记录指定 seed。

    Args:
        seed_list: 当前从 `seed.txt` 读取到的成功 seed 列表。
        episode_idx: 目标正常 episode 下标。
        seed: 准备写入该 episode 的 seed。
        overwrite: 是否允许替换已存在且不同的 seed。

    Returns:
        None.

    Raises:
        SystemExit: episode 下标为负数、会导致 `seed.txt` 中间出现空洞，
        或在未开启 `overwrite` 时与已有 seed 冲突。
    """
    if episode_idx < 0:
        raise SystemExit("episode index must be >= 0")

    if episode_idx > len(seed_list):
        raise SystemExit(
            f"Cannot write seed for episode {episode_idx}; seed.txt currently has "
            f"{len(seed_list)} entries. Use --start-episode {len(seed_list)} or fill previous episodes first."
        )

    if episode_idx < len(seed_list) and seed_list[episode_idx] != seed and not overwrite:
        raise SystemExit(
            f"seed.txt already records episode {episode_idx} as seed {seed_list[episode_idx]}, "
            f"but received seed {seed}. Use --overwrite to replace it."
        )


def record_seed(save_path, seed_list, episode_idx, seed):
    """记录一个成功 episode 的 seed，并持久化到 `seed.txt`。

    Args:
        save_path: 数据集保存目录。
        seed_list: 需要同步更新的内存 seed 列表。
        episode_idx: 要记录的正常 episode 下标。
        seed: 该 episode 对应的 seed。

    Returns:
        None。该函数会原地修改 `seed_list` 并重写 `seed.txt`。
    """
    if episode_idx == len(seed_list):
        seed_list.append(seed)
    else:
        seed_list[episode_idx] = seed
    write_seed_list(save_path, seed_list)


def existing_episode_outputs(save_path, episode_idx):
    """查找某个正常 episode 已存在的输出文件。

    Args:
        save_path: 数据集保存目录。
        episode_idx: 正常 episode 下标。

    Returns:
        该 episode 已存在的 `hdf5` 和/或 `mp4` 输出路径。
    """
    paths = [
        os.path.join(save_path, "data", f"episode{episode_idx}.hdf5"),
        os.path.join(save_path, "video", f"episode{episode_idx}.mp4"),
    ]
    return [path for path in paths if os.path.exists(path)]


def ensure_episode_writable(save_path, episode_idx, overwrite):
    """确认某个 episode 输出槽位可以写入。

    Args:
        save_path: 数据集保存目录。
        episode_idx: 正常 episode 下标。
        overwrite: 是否允许重新生成已存在的 `hdf5/mp4` 输出。

    Returns:
        None.

    Raises:
        SystemExit: 输出文件已存在且未启用 overwrite 时抛出。
    """
    existing_paths = existing_episode_outputs(save_path, episode_idx)
    if existing_paths and not overwrite:
        existing = ", ".join(existing_paths)
        raise SystemExit(
            f"Episode {episode_idx} already has saved output: {existing}. "
            "Use --overwrite to regenerate it."
        )


def close_env_safely(TASK_ENV, render_freq=0, clear_cache=False):
    """安全关闭任务环境，并在需要时关闭 viewer。

    Args:
        TASK_ENV: 任务环境实例。
        render_freq: 非 0 表示可能打开过 viewer。
        clear_cache: 是否请求清理 SAPIEN cache。

    Returns:
        None。清理过程中的异常会被忽略。
    """
    try:
        TASK_ENV.close_env(clear_cache=clear_cache)
    except Exception:
        pass

    if render_freq:
        try:
            TASK_ENV.viewer.close()
        except Exception:
            pass


def save_scene_info(args, episode_idx, info):
    """将单个 episode 的场景元信息写入 `scene_info.json`。

    Args:
        args: 运行时配置字典，使用其中的 `args["save_path"]`。
        episode_idx: episode 下标，在 JSON 中保存为 `episode_<idx>`。
        info: `TASK_ENV.play_once()` 返回的元信息，或失败记录信息。

    Returns:
        None。该函数会创建或更新 `<save_path>/scene_info.json`。
    """
    info_file_path = os.path.join(args["save_path"], "scene_info.json")

    if not os.path.exists(info_file_path):
        with open(info_file_path, "w", encoding="utf-8") as file:
            json.dump({}, file, ensure_ascii=False)

    with open(info_file_path, "r", encoding="utf-8") as file:
        info_db = json.load(file)

    info_db[f"episode_{episode_idx}"] = info

    with open(info_file_path, "w", encoding="utf-8") as file:
        json.dump(info_db, file, ensure_ascii=False, indent=4)


def get_fail_args(args):
    """构造指向失败数据集目录的配置副本。

    Args:
        args: 正常数据集的运行时配置字典。

    Returns:
        `args` 的浅拷贝，其中 `task_config` 和 `save_path` 会追加 `_fail`。
    """
    fail_args = args.copy()
    fail_args["task_config"] = f"{args['task_config']}_fail"
    fail_args["save_path"] = f"{args['save_path']}_fail"
    return fail_args


def reserve_fail_episode(args, seed):
    """为某个失败 seed 预留下一个失败 episode 下标。

    Args:
        args: 正常数据集的运行时配置字典。
        seed: 需要记录的失败 seed。

    Returns:
        三元组 `(fail_args, fail_seed_list, fail_episode_idx)`。

    Raises:
        SystemExit: 下一个失败 episode 已经存在输出文件时抛出。
    """
    fail_args = get_fail_args(args)
    os.makedirs(fail_args["save_path"], exist_ok=True)
    fail_seed_list = read_seed_list(fail_args["save_path"])
    fail_episode_idx = len(fail_seed_list)
    ensure_episode_writable(fail_args["save_path"], fail_episode_idx, overwrite=False)
    return fail_args, fail_seed_list, fail_episode_idx


def build_failure_info(info, seed, reason, success=False):
    """为场景信息追加失败记录字段。

    Args:
        info: 环境返回的场景信息。非字典类型会被替换为空字典。
        seed: 产生失败轨迹的 seed。
        reason: 描述失败原因的异常对象或消息。
        success: 失败轨迹重录时是否意外通过成功检查。

    Returns:
        追加了 `seed`、`success` 和 `failure_reason` 的 info 副本。
    """
    if not isinstance(info, dict):
        info = {}
    info = info.copy()
    info["seed"] = seed
    info["success"] = success
    info["failure_reason"] = str(reason)
    return info


def save_traj_data_safely(TASK_ENV, save_path, episode_idx):
    """保存轨迹数据，并在结束后恢复环境原始保存目录。

    Args:
        TASK_ENV: 任务环境实例。
        save_path: 目标数据集目录。
        episode_idx: 用于 `_traj_data/episodeX.pkl` 的 episode 下标。

    Returns:
        None。保存失败时只打印 warning，不继续向外抛出。
    """
    original_save_dir = getattr(TASK_ENV, "save_dir", None)
    try:
        TASK_ENV.save_dir = save_path
        TASK_ENV.save_traj_data(episode_idx)
    except Exception as e:
        print(f"\033[93mWarning: failed to save traj data for failed episode {episode_idx}: {e}\033[0m")
    finally:
        if original_save_dir is not None:
            TASK_ENV.save_dir = original_save_dir


def remove_data_cache_safely(TASK_ENV):
    """如果环境中存在 cache 路径，则删除缓存帧 pkl 文件。

    Args:
        TASK_ENV: 任务环境实例。

    Returns:
        None。cache 路径不存在或清理失败都会被容忍。
    """
    if not hasattr(TASK_ENV, "folder_path") or "cache" not in TASK_ENV.folder_path:
        return

    try:
        TASK_ENV.remove_data_cache()
    except Exception as e:
        print(f"\033[93mWarning: failed to remove data cache: {e}\033[0m")


def finalize_hdf5_episode(TASK_ENV, target_args, target_episode_idx, clear_cache, require_cache=True):
    """关闭 episode，并将缓存帧合并为 HDF5 和视频。

    Args:
        TASK_ENV: 需要 finalize cache 的任务环境实例。
        target_args: 目标数据集目录对应的运行时配置。
        target_episode_idx: 输出为 `episodeX` 时使用的 episode 下标。
        clear_cache: 关闭环境时是否请求清理 SAPIEN cache。
        require_cache: 为 True 时，缓存合并失败会继续抛出异常；为 False
            时只打印 warning，主要用于失败 episode 的归档。

    Returns:
        None。函数结束前会恢复环境原始的 `save_dir` 和 `ep_num`。
    """
    original_save_dir = getattr(TASK_ENV, "save_dir", None)
    original_ep_num = getattr(TASK_ENV, "ep_num", None)
    closed = False

    try:
        TASK_ENV.save_dir = target_args["save_path"]
        TASK_ENV.ep_num = target_episode_idx
        TASK_ENV.close_env(clear_cache=clear_cache)
        closed = True

        try:
            TASK_ENV.merge_pkl_to_hdf5_video()
        except Exception as e:
            if require_cache:
                raise
            print(f"\033[93mWarning: failed to merge failed episode {target_episode_idx}: {e}\033[0m")
        finally:
            remove_data_cache_safely(TASK_ENV)
    finally:
        if not closed:
            close_env_safely(TASK_ENV, clear_cache=clear_cache)
        if original_save_dir is not None:
            TASK_ENV.save_dir = original_save_dir
        if original_ep_num is not None:
            TASK_ENV.ep_num = original_ep_num


def mark_failure_saved(error):
    """标记异常，避免外层处理器重复保存失败数据。

    Args:
        error: 需要标记的异常对象。

    Returns:
        同一个异常对象，并附加 `failure_data_saved=True` 属性。
    """
    error.failure_data_saved = True
    return error


def save_failed_recording_for_seed(TASK_ENV, args, seed, reason, render_freq=0):
    """尝试将失败 seed 的轨迹记录到 `_fail` 数据集。

    Args:
        TASK_ENV: 可复用的任务环境实例。
        args: 正常数据集的运行时配置字典。
        seed: 需要重放并归档的失败 seed。
        reason: 原始失败异常或失败原因消息。
        render_freq: 重放失败 seed 时使用的渲染频率。

    Returns:
        None。函数会尽最大努力将失败数据、场景信息、seed 列表、HDF5
        和视频写入 `<save_path>_fail`。
    """
    try:
        fail_args, fail_seed_list, fail_episode_idx = reserve_fail_episode(args, seed)
        record_args = fail_args.copy()
        record_args["need_plan"] = True
        record_args["save_data"] = True
        record_args["render_freq"] = render_freq
        record_args.pop("left_joint_path", None)
        record_args.pop("right_joint_path", None)

        info, record_error = {}, None
        print(f"\033[93mSave failed trajectory to: {fail_args['save_path']} / episode{fail_episode_idx}\033[0m")

        TASK_ENV.setup_demo(now_ep_num=fail_episode_idx, seed=seed, **record_args)
        try:
            info = TASK_ENV.play_once()
        except Exception as e:
            record_error = e

        try:
            success = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
        except Exception:
            success = False

        save_traj_data_safely(TASK_ENV, fail_args["save_path"], fail_episode_idx)
        save_scene_info(fail_args, fail_episode_idx, build_failure_info(info, seed, reason, success=success))
        finalize_hdf5_episode(TASK_ENV, fail_args, fail_episode_idx, clear_cache=False, require_cache=False)
        record_seed(fail_args["save_path"], fail_seed_list, fail_episode_idx, seed)

        if record_error is not None:
            print(f"\033[93mWarning: failed trajectory recording stopped early: {record_error}\033[0m")
    except Exception as e:
        close_env_safely(TASK_ENV, render_freq=render_freq)
        print(f"\033[93mWarning: failed to save failed trajectory for seed {seed}: {e}\033[0m")


def collect_pre_motion_for_seed(TASK_ENV, args, episode_idx, seed, render_freq):
    """为单个 seed 执行运动规划，并只保存关节轨迹。

    Args:
        TASK_ENV: 任务环境实例。
        args: 运行时配置字典。该函数会修改 `need_plan`、`save_data`、
            `render_freq`，并移除旧的关节轨迹字段。
        episode_idx: 正常 episode 下标。
        seed: 用于初始化任务环境的 seed。
        render_freq: 规划阶段 viewer 的渲染频率。

    Returns:
        None。规划和成功检查通过后，写入 `_traj_data/episodeX.pkl`。

    Raises:
        RuntimeError: 规划失败或任务成功检查失败时抛出。
        Exception: 环境初始化或执行过程中的异常会在清理后继续向外抛出。
    """
    args["need_plan"] = True
    args["save_data"] = False
    args["render_freq"] = render_freq
    args.pop("left_joint_path", None)
    args.pop("right_joint_path", None)

    try:
        TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed, **args) # 初始化环境
        TASK_ENV.play_once() # 执行一次，进行运动规划

        if not (TASK_ENV.plan_success and TASK_ENV.check_success()): # 失败的判断条件
            raise RuntimeError(f"Planning failed for episode {episode_idx}, seed {seed}")

        TASK_ENV.save_traj_data(episode_idx) # 保存轨迹数据
    finally:
        close_env_safely(TASK_ENV, render_freq=render_freq) # 安全关闭环境


def collect_hdf5_for_seed(TASK_ENV, args, episode_idx, seed, clear_cache):
    """回放已规划轨迹，并为单个 seed 保存 HDF5 和视频数据。

    Args:
        TASK_ENV: 任务环境实例。
        args: 运行时配置字典。该函数会修改 `need_plan`、`save_data`、
            `render_freq`、`left_joint_path` 和 `right_joint_path`。
        episode_idx: 正常 episode 下标。
        seed: 用于复现同一任务场景的 seed。
        clear_cache: 关闭 episode 时是否清理 SAPIEN cache。

    Returns:
        None。成功时会写入 `scene_info.json`、`data/episodeX.hdf5`
        和 `video/episodeX.mp4`。

    Raises:
        Exception: 回放、成功检查或必要的缓存合并失败时抛出。可行时会先保存
            失败数据，再继续抛出异常。
    """
    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = True

    finalized = False
    try:
        TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed, **args)

        traj_data = TASK_ENV.load_tran_data(episode_idx) # 加载轨迹数据
        args["left_joint_path"] = traj_data["left_joint_path"]
        args["right_joint_path"] = traj_data["right_joint_path"]
        TASK_ENV.set_path_lst(args)

        info, play_error = {}, None
        try:
            info = TASK_ENV.play_once() # 指定self.move时直接执行轨迹回放，不再调用规划器进行规划
        except Exception as e:
            play_error = e

        check_error = None
        try:
            success = play_error is None and TASK_ENV.plan_success and TASK_ENV.check_success() # 成功的判断条件
        except Exception as e:
            check_error = e
            success = False

        if success: # 收集成功轨迹
            save_scene_info(args, episode_idx, info) # 保存场景信息
            finalize_hdf5_episode(TASK_ENV, args, episode_idx, clear_cache=clear_cache, require_cache=True) # 对一个成功 episode 做收尾，把采集过程中缓存的 pkl 帧数据合并成最终的 HDF5 和 mp4，并清理缓存
            finalized = True
            return

        reason = play_error or check_error or "Collect Error"
        fail_args, fail_seed_list, fail_episode_idx = reserve_fail_episode(args, seed) # 预留失败轨迹
        print(f"\033[93mSave failed trajectory to: {fail_args['save_path']} / episode{fail_episode_idx}\033[0m")
        save_traj_data_safely(TASK_ENV, fail_args["save_path"], fail_episode_idx)
        save_scene_info(fail_args, fail_episode_idx, build_failure_info(info, seed, reason, success=False))
        finalize_hdf5_episode(TASK_ENV, fail_args, fail_episode_idx, clear_cache=clear_cache, require_cache=False)
        finalized = True
        record_seed(fail_args["save_path"], fail_seed_list, fail_episode_idx, seed) # 把当前失败的 seed 记录到失败目录的 seed.txt 里

        if play_error is not None:
            raise mark_failure_saved(play_error)
        if check_error is not None:
            raise mark_failure_saved(check_error)
        raise mark_failure_saved(AssertionError("Collect Error"))
    finally:
        if not finalized: # 如果没有正确结束，需要再次关闭环境
            close_env_safely(TASK_ENV, clear_cache=clear_cache)


def generate_episode_instructions(args):
    """为已采集 episode 生成语言指令。

    Args:
        args: 运行时配置字典，使用其中的 `task_name`、`task_config`
            和 `language_num`。

    Returns:
        None。通过 `os.system` 执行 `description/gen_episode_instructions.sh`。
    """
    command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
    os.system(command)


def collect_data_from_given_seeds(TASK_ENV, args, seeds, start_episode=None, overwrite=False):
    """为显式指定的 seeds 采集完整数据。

    Args:
        TASK_ENV: 任务环境实例。
        args: 目标数据集的运行时配置字典。
        seeds: 从命令行传入的有序 seed 列表。
        start_episode: 第一个成功 seed 写入的正常 episode 下标。未指定时，
            会追加到已有 `seed.txt` 之后。
        overwrite: 是否允许替换已有的正常输出和 seed 记录。

    Returns:
        None。成功 seed 写入正常数据集；失败 seed 归档到 `_fail` 数据集
        并跳过。
    """
    seed_list = read_seed_list(args["save_path"])
    if start_episode is None:
        start_episode = len(seed_list)

    if start_episode < 0:
        raise SystemExit("--start-episode must be >= 0")

    plan_render_freq = args.get("render_freq", 0)
    clear_cache_freq = args.get("clear_cache_freq", 1)
    fail_num = 0
    success_num = 0
    failed_seeds = []
    episode_idx = start_episode

    print("\033[93m" + "[Start Data Collection From Given Seeds]" + "\033[0m")
    print(f"Start episode: {start_episode}, Seeds: {seeds}")

    for seed in seeds:
        check_seed_slot(seed_list, episode_idx, seed, overwrite)
        ensure_episode_writable(args["save_path"], episode_idx, overwrite)

        try:
            print(f"\033[34mTask name: {args['task_name']}, episode: {episode_idx}, seed: {seed}\033[0m")
            collect_pre_motion_for_seed(TASK_ENV, args, episode_idx, seed, plan_render_freq) # plan 规划
            collect_hdf5_for_seed(
                TASK_ENV,
                args,
                episode_idx,
                seed,
                clear_cache=((episode_idx + 1) % clear_cache_freq == 0),
            ) # replay 轨迹并保存数据
            record_seed(args["save_path"], seed_list, episode_idx, seed)
            print(f"collect data episode {episode_idx} success! (seed = {seed})")
            episode_idx += 1
            success_num += 1
        except UnStableError as e:
            print(" -------------")
            print(f"collect data episode {episode_idx} fail! (seed = {seed})")
            print("Error: ", e)
            print(" -------------")
            if not getattr(e, "failure_data_saved", False):
                save_failed_recording_for_seed(TASK_ENV, args, seed, e, render_freq=plan_render_freq)
            fail_num += 1
            failed_seeds.append(seed)
            time.sleep(0.3)
            continue
        except Exception as e:
            print(" -------------")
            print(f"collect data episode {episode_idx} fail! (seed = {seed})")
            print("Error: ", e)
            print(traceback.format_exc())
            print(" -------------")
            if not getattr(e, "failure_data_saved", False):
                save_failed_recording_for_seed(TASK_ENV, args, seed, e, render_freq=plan_render_freq)
            fail_num += 1
            failed_seeds.append(seed)
            time.sleep(1)
            continue

    if fail_num:
        print(
            f"\033[93mSkipped {fail_num} failed seed(s), collected {success_num} / {len(seeds)} given seed(s). "
            f"Failed seeds: {failed_seeds}\033[0m"
        )

    generate_episode_instructions(args)


def collect_seed_and_pre_motion(TASK_ENV, args):
    """为非显式 seed 模式准备 seed 列表和预规划轨迹。

    Args:
        TASK_ENV: 任务环境实例。
        args: 运行时配置字典。如果 `use_seed` 为 false，该函数会修改规划
            相关字段，并写入 `seed.txt` 和 `_traj_data`；如果 `use_seed`
            为 true，则只读取 `seed.txt`。

    Returns:
        按正常 episode 下标排序的 seed 列表。
    """
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    if args["use_seed"]:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]
        return seed_list

    print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
    args["need_plan"] = True

    if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
        seed_list = read_seed_list(args["save_path"])
        if len(seed_list) != 0:
            suc_num = len(seed_list)
            epid = max(seed_list) + 1
        print(f"Exist seed file, Start from: {epid} / {suc_num}")

    while suc_num < args["episode_num"]:
        try:
            need_save_failed_recording = False
            TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
            TASK_ENV.play_once()

            if TASK_ENV.plan_success and TASK_ENV.check_success():
                print(f"simulate data episode {suc_num} success! (seed = {epid})")
                seed_list.append(epid)
                TASK_ENV.save_traj_data(suc_num)
                suc_num += 1
            else:
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                need_save_failed_recording = True
                fail_num += 1

            TASK_ENV.close_env()

            if args["render_freq"]:
                TASK_ENV.viewer.close()

            if need_save_failed_recording:
                save_failed_recording_for_seed(
                    TASK_ENV,
                    args,
                    epid,
                    "Pre motion planning/check_success failed",
                    render_freq=args["render_freq"],
                )
        except UnStableError as e:
            print(" -------------")
            print(f"simulate data episode {suc_num} fail! (seed = {epid})")
            print("Error: ", e)
            print(" -------------")
            fail_num += 1
            close_env_safely(TASK_ENV, render_freq=args["render_freq"])

            save_failed_recording_for_seed(TASK_ENV, args, epid, e, render_freq=args["render_freq"])
            time.sleep(0.3)
        except Exception as e:
            print(" -------------")
            print(f"simulate data episode {suc_num} fail! (seed = {epid})")
            print("Error: ", e)
            print(" -------------")
            fail_num += 1
            close_env_safely(TASK_ENV, render_freq=args["render_freq"])

            save_failed_recording_for_seed(TASK_ENV, args, epid, e, render_freq=args["render_freq"])
            time.sleep(1)

        epid += 1
        write_seed_list(args["save_path"], seed_list)

    print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    return seed_list


def collect_data_from_seed_list(TASK_ENV, args, seed_list):
    """基于已有有序 seed 列表采集 HDF5 和视频数据。

    Args:
        TASK_ENV: 任务环境实例。
        args: 运行时配置字典，使用其中的 `episode_num`、`collect_data`、
            `clear_cache_freq` 和 `save_path`。
        seed_list: 按正常 episode 下标排序的 seed 列表。

    Returns:
        None。已存在的 HDF5 episode 会被跳过；失败 seed 会归档到 `_fail`，
        然后继续采集后续 episode。
    """
    if not args["collect_data"]:
        return

    print("\033[93m" + "[Start Data Collection]" + "\033[0m")

    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = True

    clear_cache_freq = args["clear_cache_freq"]

    st_idx = 0

    def exist_hdf5(idx):
        file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
        return os.path.exists(file_path)

    while exist_hdf5(st_idx):
        st_idx += 1

    failed_seeds = []
    for episode_idx in range(st_idx, args["episode_num"]):
        print(f"\033[34mTask name: {args['task_name']}\033[0m")

        try:
            collect_hdf5_for_seed(
                TASK_ENV,
                args,
                episode_idx,
                seed_list[episode_idx],
                clear_cache=((episode_idx + 1) % clear_cache_freq == 0),
            )
        except Exception as e:
            if not getattr(e, "failure_data_saved", False):
                save_failed_recording_for_seed(TASK_ENV, args, seed_list[episode_idx], e, render_freq=0)
            failed_seeds.append(seed_list[episode_idx])
            print(
                f"\033[93mSkip failed seed {seed_list[episode_idx]} "
                f"for episode {episode_idx}, continue collecting next seed.\033[0m"
            )
            time.sleep(1)
            continue

    if failed_seeds:
        print(f"\033[93mSkipped failed seeds during data collection: {failed_seeds}\033[0m")

    generate_episode_instructions(args) # 生成语言指令


def main(task_name=None, task_config=None, seeds=None, start_episode=None, overwrite=False, episode_num=None):
    """加载配置、解析 embodiment 文件，并启动数据采集。

    Args:
        task_name: `envs` 目录下的任务环境名称。
        task_config: `task_config` 目录下的 YAML 配置名，不包含 `.yml`。
        seeds: 命令行传入的可选显式 seed 列表。
        start_episode: 显式 seed 模式下可选的首个正常 episode 下标。
        overwrite: 显式 seed 采集时是否允许覆盖已有输出。

    Returns:
        None。该函数会构造运行时 `args` 字典，并交给 `run()` 执行。
    """

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if episode_num is not None:
        if episode_num <= 0:
            raise ValueError("--episode-num must be positive")
        args["episode_num"] = episode_num

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # 打印当前采集配置
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    args["given_seeds"] = seeds or []
    args["start_episode"] = start_episode
    args["overwrite"] = overwrite
    run(task, args)  # 数据收集主程序


def run(TASK_ENV, args):
    """根据是否提供显式 seeds 分发数据采集流程。

    Args:
        TASK_ENV: 任务环境实例。
        args: 已完整解析的运行时配置字典。

    Returns:
        None。该函数会创建数据集目录，并执行显式 seed 采集，或执行配置
        驱动的 seed 准备和数据采集流程。
    """
    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    os.makedirs(args["save_path"], exist_ok=True)

    if args.get("given_seeds"): # 显式指定 seed 模式，直接采集数据并跳过 seed 准备阶段
        collect_data_from_given_seeds(
            TASK_ENV,
            args,
            args["given_seeds"],
            start_episode=args.get("start_episode"),
            overwrite=args.get("overwrite", False),
        )
    else: # 自动生成seed模式，先准备 seed 和预规划轨迹，再采集数据
        seed_list = collect_seed_and_pre_motion(TASK_ENV, args) 
        collect_data_from_seed_list(TASK_ENV, args, seed_list)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("seeds", nargs="*", help="Seed values, e.g. 123 456 or 123,456")
    parser.add_argument("--seed", dest="seed_options", action="append", default=[], help="Seed value; can be used multiple times")
    parser.add_argument("--start-episode", type=int, default=None, help="Episode index for the first given seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing saved output for selected episodes")
    parser.add_argument("--episode-num", type=int, default=None, help="Override episode_num in task config")
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config
    seeds = parse_seed_values(parser.seed_options + parser.seeds)

    main(task_name=task_name, task_config=task_config, seeds=seeds, start_episode=parser.start_episode,
         overwrite=parser.overwrite, episode_num=parser.episode_num)
