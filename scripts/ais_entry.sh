#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
    gpu-check)
        shift
        exec python3 -u scripts/ais_gpu_check.py "$@"
        ;;
    vm-check)
        shift
        exec python -u scripts/ais_vm_check.py "$@"
        ;;
    desktop-service)
        shift
        exec python3 -u scripts/ais_desktop_service.py "$@"
        ;;
    sft-audit)
        shift
        for option in "$@"; do
            if [[ "$option" == "--full" ]]; then
                exec python3 -u scripts/ais_sft_runtime.py "$@"
            fi
        done
        exec python3 -u scripts/ais_sft_audit.py "$@"
        ;;
    sft-eval)
        shift
        exec python3 -u scripts/ais_sft_eval.py "$@"
        ;;
    sft-perf)
        shift
        exec python3 -u scripts/ais_sft_perf.py "$@"
        ;;
    sft-episode)
        shift
        exec "${CUA_MODEL_PYTHON:-python3}" -u scripts/ais_sft_episode.py "$@"
        ;;
    frontier)
        shift
        exec python3 -u scripts/ais_frontier.py "$@"
        ;;
    frontier-server)
        shift
        exec python3 -u scripts/ais_frontier_server.py "$@"
        ;;
    train-settings)
        shift
        exec python3 -u scripts/ais_gpu_check.py --settings "$@"
        ;;
    train-check)
        shift
        exec "${CUA_TRAIN_PYTHON:-python}" -u scripts/runtime_env.py scripts/smoke_training_runtime.py --ais "$@"
        ;;
    train)
        shift
        if [[ "${WORLD_SIZE:-1}" == 8 ]]; then
            exec "${CUA_TRAIN_PYTHON:-python}" -u scripts/runtime_env.py scripts/ais_distributed_rl.py "$@"
        fi
        train=("${CUA_TRAIN_PYTHON:-python}" -u scripts/runtime_env.py scripts/launch_local_rl.py --ais-pilot "$@")
        if [[ -n "${CUA_CONTROLLER_URL:-}" ]]; then
            exec "${train[@]}"
        fi
        exec python3 -u scripts/ais_rl_job.py "${train[@]}"
        ;;
    *)
        echo 'Usage: ais_entry.sh {gpu-check|vm-check|desktop-service|sft-audit|sft-eval|sft-perf|sft-episode|frontier|frontier-server|train-settings|train-check|train} [arguments...]' >&2
        exit 2
        ;;
esac
