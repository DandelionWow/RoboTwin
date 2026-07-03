#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: bash collect_data.sh <task_name> <task_config> <gpu_id> [--denoiser oidn|optix|none] [--oidn-library-dir /path/to/oidn_library]"
    exit 2
fi

task_name=${1}
task_config=${2}
gpu_id=${3}
shift 3

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES="${gpu_id}"

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py "${task_name}" "${task_config}" "$@"
status=$?
rm -rf "data/${task_name}/${task_config}/.cache"
exit "${status}"
