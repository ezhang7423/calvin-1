
import argparse
from collections import Counter, defaultdict
import logging
from multiprocessing import shared_memory
import os
from pathlib import Path
import pickle
import random
import sys
import time


# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint


import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm


from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000



def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class CustomModel(CalvinBaseModel):
    def __init__(self):
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        raise NotImplementedError

    def reset(self):
        """
        This is called
        """
        raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        raise NotImplementedError


def evaluate_policy(model, env, epoch, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    if rollout_freq:
        try:
            existing_shm = shared_memory.SharedMemory(name='eval_seq')
            preloaded_sequences = pickle.loads(existing_shm.buf)
        except FileNotFoundError:
            preloaded_sequences = torch.load(Path(os.environ["DATA_GRAND_CENTRAL"]) / "eval_sequences.pt")
            random.shuffle(preloaded_sequences)
            
        eval_sequences = preloaded_sequences[:NUM_SEQUENCES]
    else:
        eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    # robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    if rollout_freq:
        robot_obs, scene_obs, task_goal_image = initial_state
        if eval_sequence[0] != "place_in_drawer" and eval_sequence[0] != "place_in_slider":
            robot_obs = None
    else:
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        task_goal_image = None    
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, task_goal_image=task_goal_image)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, task_goal_image=None):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]


    if task_goal_image is not None:
        goal = cuda_slice(task_goal_image, rollout_freq, device=model.device)
    else:
        goal = lang_annotation
    
    goal_step = rollout_freq
    goal_i = 0
    model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):        
        # action = model.step(obs, lang_annotation)
        resample_goal: bool = (step + rollout_freq) < 128 and not ((step + 1) % rollout_freq)
        if task_goal_image is not None and resample_goal:
            goal_i += 1
            if goal_i % 2 == 0:
                goal_step += rollout_freq
            goal = cuda_slice(task_goal_image, goal_step, device=model.device)
            lk = goal['rgb_obs']['rgb_static'].squeeze().cpu().permute(1, 2,
 0).numpy()
            join_vis_lang(lk, f'goal: step={goal_step} ', name='goal')

        action = model.step(obs, goal)        
        obs, _, _, current_info = env.step(action)
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, f'{subtask} {step=} {resample_goal=}')
            if resample_goal:
                print(f'resampling goal! {step=}')
                time.sleep(3)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
    return False


def cuda_slice(d, slice, device=None):
    if isinstance(d, np.ndarray):
        d = torch.from_numpy(d)
    if isinstance(d, torch.Tensor):
        if device is None:
            return d.cuda()[slice].permute(2, 0, 1)[None, None]
        else:
            return d.to(device)[slice].permute(2, 0, 1)[None, None]
    if isinstance(d, dict):
        newb = {}
        for k in d:
            newb[k] = cuda_slice(d[k], slice, device)
    if isinstance(d, list):
        newb = []
        for i in range(len(d)):
            newb.append(cuda_slice(d[i], slice, device))
    return newb

def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    from os.path import join as j
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.", default=j(os.environ['DATA_GRAND_CENTRAL'], 'task_D_D') )

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir.", default='/home/ubuntu/data/runs/2023-07-24/10-14-58'
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='/home/ubuntu/data/runs/2023-07-24/10-14-58/saved_models/epoch=5.ckpt',
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.",)

    parser.add_argument("--eval_log_dir", default=j(os.environ['DATA_GRAND_CENTRAL'], 'gcbc_results'), type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()
    
    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = checkpoint.stem.split("=")[1]
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


if __name__ == "__main__":
    rollout_freq = 16
    
    main()
