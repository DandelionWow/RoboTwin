import json
import os


def generate_episode_instructions_from_save_path(args):
    from description.utils import generate_episode_instructions as instruction_utils

    original_load_scene_info = instruction_utils.load_scene_info
    original_save_episode_descriptions = instruction_utils.save_episode_descriptions

    def patched_load_scene_info(task_name, setting, scene_info_path):
        file_path = os.path.join(args["save_path"], "scene_info.json")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"\033[1mERROR: Scene info file '{file_path}' not found.\033[0m")
            exit(1)
        except json.JSONDecodeError:
            print(f"\033[1mERROR: Scene info file '{file_path}' contains invalid JSON.\033[0m")
            exit(1)

    def patched_save_episode_descriptions(task_name, setting, generated_descriptions, save_path):
        output_dir = os.path.join(args["save_path"], "instructions")
        os.makedirs(output_dir, exist_ok=True)

        for episode_desc in generated_descriptions:
            episode_index = episode_desc["episode_index"]
            output_file = os.path.join(output_dir, f"episode{episode_index}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "seen": episode_desc.get("seen", []),
                        "unseen": episode_desc.get("unseen", []),
                    },
                    f,
                    indent=2,
                )

    try:
        instruction_utils.load_scene_info = patched_load_scene_info
        instruction_utils.save_episode_descriptions = patched_save_episode_descriptions

        setting = args["task_config"]
        scene_info = instruction_utils.load_scene_info(args["task_name"], setting, args["save_path"])
        episodes = instruction_utils.extract_episodes_from_scene_info(scene_info)
        descriptions = instruction_utils.generate_episode_descriptions(
            args["task_name"],
            episodes,
            args["language_num"],
        )
        instruction_utils.save_episode_descriptions(
            args["task_name"],
            setting,
            descriptions,
            args["save_path"],
        )
    finally:
        instruction_utils.load_scene_info = original_load_scene_info
        instruction_utils.save_episode_descriptions = original_save_episode_descriptions
