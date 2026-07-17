#!/usr/bin/env python3
import os, glob
from pathlib import Path
os.environ.pop("SQUEUE_FORMAT", None)
from simple_slurm import Slurm


# Equivalent to workflow.launchDir
LAUNCH_DIR = Path(os.getcwd())
LOG_DIR = LAUNCH_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def array_string(ids, max_running=None):

    array_arg = ",".join(map(str, ids)) 
    if max_running is not None:
        array_arg += f"%{max_running}"
    return array_arg

def add_environment_variables(command):

    command = f"""
    export WANDB_API_KEY=$(cat {os.path.expanduser('~/.secrets/wandb-api-key')}); \\
    export WANDB_RUN_GROUP="slurm-$SLURM_ARRAY_JOB_ID"; \\
    """ + command
    return command

def get_logs(name):

    return {"output": str(LOG_DIR / f"{name}.%A_%a.out"), "error": str(LOG_DIR / f"{name}.%A_%a.err")}


def clear_logs(name):

    log_dict = get_logs(name)
    logs_stdout = glob.glob(log_dict["output"].replace("%A_%a", "*"))
    logs_stderr = glob.glob(log_dict["error"].replace("%A_%a", "*"))
    n_removed = 0
    for log in logs_stdout + logs_stderr:
        if os.path.exists(log):
            os.remove(log)  
            n_removed += 1
    print(f'Removed {n_removed} logs')

def submit_job(slurm_args, name, command):

    slurm_args["job_name"] = name
    slurm_args.update(get_logs(name))
    slurm = Slurm(**slurm_args)

    print(f'----------------- Submitting {name}')
    print(f'Clearing logs {name}')
    clear_logs(name)
    print(f'Command:\n{command}')
    job_id = slurm.sbatch(command)
    print(f"Submitted {name} jobid: {job_id}")
    return job_id



########################################################################################
########################################################################################
##
##
## Slurm arguments
##
##
########################################################################################
########################################################################################



SLURM_ARGS = {
        # "cpus_per_task": 96,
        # "mem_per_cpu": "1950M",
        "nodes": 1,
        "exclusive": False,
        "mem_per_cpu": 2900,
        "cpus_per_task": 24,
        "time": "12:00:00",
        "partition": "normal",
        "gres": "gpu:1",
        "account": "a0186",
        "ntasks_per_node": 1,
    }


########################################################################################
########################################################################################
##
##
## Jobs submission definitions
##
##
########################################################################################
########################################################################################


def execute_biflow_demo(
    *,
    name,
    submit=False,
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = SLURM_ARGS | kwargs


    command = f"""
    set -euo pipefail; \\
    srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
            bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
            uv run repos/autoregressive-transformers-demos/bidirectional_flow_blob_demo.py \\
            --wandb --wandb-run-name={name} --wandb-entity=cosmogridcollab \\
            --ae-epochs=30 --flow-epochs=500 \\
            --checkpoint-ae=toy_flow_outputs/toy_ae_checkpoint.pt' 
    """

    command = add_environment_variables(command)
    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None

    return job_id


def execute_autoreg_demo(
    *,
    name,
    submit=False,
    dataset_type="deterministic",
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = SLURM_ARGS | kwargs


    if submit:
        command = f"""
        set -euo pipefail; \\
        srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
                uv run repos/autoregressive-transformers-demos/continuous_patch_ar_demo.py \\
                --wandb-run-name={name} --wandb-entity=cosmogridcollab --dataset-type={dataset_type}
                --checkpoint-tokenizer=autoreg_outputs/continuous_tokenizer.pt' 
        """
    else:
        command = f"""
        uv run repos/autoregressive-transformers-demos/continuous_patch_ar_demo.py --wandb-run-name={name} --wandb-entity=cosmogridcollab --device=cuda --dataset-type={dataset_type} --checkpoint-tokenizer=autoreg_outputs/continuous_tokenizer.pt
        """


    command = add_environment_variables(command)
    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None

    return job_id

def execute_flexible_autoreg_demo(
    *,
    name,
    submit=False,
    dataset_type="deterministic",
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = SLURM_ARGS | kwargs

    if submit:
        command = f"""
        set -euo pipefail; \\
        srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
                uv run repos/autoregressive-transformers-demos/flexible_prompt_patch_ar_demo.py \\
                --wandb-run-name={name} --wandb-entity=cosmogridcollab --dataset-type={dataset_type}
        """

        command = add_environment_variables(command)

    else:
        command = f"""
        uv run repos/autoregressive-transformers-demos/flexible_prompt_patch_ar_demo.py --wandb-run-name={name} --wandb-entity=cosmogridcollab --device=cuda --dataset-type={dataset_type}
        """

    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None

    return job_id


def execute_flexible_latent_autoreg_demo(
    *,
    name,
    submit=False,
    ar_loss="latent",
    dataset_type="deterministic",
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = SLURM_ARGS | kwargs

    if submit:
        command = f"""
        set -euo pipefail; \\
        srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
                uv run repos/autoregressive-transformers-demos/flexible_prompt_latent_ar_demo.py \\
                --run-name={name} --wandb-entity=cosmogridcollab --ar-loss={ar_loss} --dataset-type={dataset_type}'
        """

        command = add_environment_variables(command)

    else:
        command = f"""
        uv run repos/autoregressive-transformers-demos/flexible_prompt_latent_ar_demo.py\\
                 --run-name={name} --wandb-entity=cosmogridcollab --device=cuda --ar-loss={ar_loss} --dataset-type={dataset_type}
        """

    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None

    return job_id









def execute_autoregressive_pyramid_demo(
    *,
    name,
    submit=False,
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = SLURM_ARGS | kwargs

    if submit:
        command = f"""
        set -euo pipefail; \\
        srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
                uv run repos/autoregressive-transformers-demos/autoregressive_pyramid_demo.py \\
                --run-name={name} --wandb-entity=cosmogridcollab'
        """

        command = add_environment_variables(command)

    else:
        command = f"""
        uv run repos/autoregressive-transformers-demos/autoregressive_pyramid_demo.py\\
                 --run-name={name} --wandb-entity=cosmogridcollab --device=cuda
        """

    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None

    return job_id






def execute_autoregressive_pyramid_demo(
    *,
    name,
    submit=False,
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = SLURM_ARGS | kwargs

    if submit:
        command = f"""
        set -euo pipefail; \\
        srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
                uv run repos/autoregressive-transformers-demos/gaussian_operator_superres.py  --stage=all\\
                --wandb-run-name={name} --wandb-entity=cosmogridcollab'
        """

        command = add_environment_variables(command)

    else:
        command = f"""
        uv run repos/autoregressive-transformers-demos/gaussian_operator_superres.py --stage=all\\
                 --wandb-run-name={name} --wandb-entity=cosmogridcollab 
        """

    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None

    return job_id
########################################################################################
########################################################################################
##
##
## Pipeline
##
##
########################################################################################
########################################################################################


execute_biflow_demo(name="//demo_blob_flow", 
             submit=True)

execute_autoreg_demo(name="//demo_autoreg", 
             submit=False)

execute_flexible_autoreg_demo(name="//demo_flexible_autoreg", 
             submit=False)

execute_flexible_latent_autoreg_demo(name="//demo_flexible_latent_autoreg_loss_latent", 
                                     ar_loss="latent",
                                     dataset_type="deterministic",
                                     submit=True)

execute_flexible_latent_autoreg_demo(name="//demo_flexible_latent_autoreg_loss_decoderaware", 
                                     ar_loss="decoder-aware",
                                     dataset_type="deterministic",
                                     submit=True)


execute_flexible_latent_autoreg_demo(name="//demo_flexible_latent_autoreg_loss_latent_nondeterministic", 
                                     ar_loss="latent",
                                     dataset_type="nondeterministic",
                                     submit=True)

execute_flexible_latent_autoreg_demo(name="//demo_flexible_latent_autoreg_loss_decoderaware_nondeterministic", 
                                     ar_loss="decoder-aware",
                                     dataset_type="nondeterministic",
                                     submit=True)

execute_autoregressive_pyramid_demo(name="demo_autoregressive_pyramid_highlr_batchsize512", 
                                    submit=True)