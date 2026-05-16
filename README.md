# SpiderBot Training Environment

## Overview

This repository contains the Isaac Lab / Isaac Sim project for the SpiderBot training environment. It includes the environment definition, task configuration, agents, and extension wrapper needed to run training and evaluation in Isaac Lab.

This project is intended to be used as a development starting point for:
- reinforcement learning experiments,
- robot environment configuration,
- training with Isaac Sim / Omniverse,
- embedding custom robot models and reward logic.

## Model Visuals

Use this section to add visual references for your robot design.

### SolidWorks Model

![SolidWorks Model](path/to/solidworks_model_image.png)

*Caption: SolidWorks CAD model of the SpiderBot.*

### URDF Model

![URDF Model](path/to/urdf_model_image.png)

*Caption: URDF visualization for the robot used in simulation.*

> Replace `path/to/...` with the actual image path or URL for your repository.

## Environment Configuration

The environment configuration is defined in `source/SpiderBotTraining_1/SpiderBotTraining_1/tasks/direct/spiderbottraining_1/spiderbottraining_1_env_cfg.py` and the environment implementation is in `source/SpiderBotTraining_1/SpiderBotTraining_1/tasks/direct/spiderbottraining_1/spiderbottraining_1_env.py`.

### Observation Space

The primary observation data includes:
- robot state: joint positions, joint velocities, base pose, and base velocity
- target or goal state: desired position or orientation for the task
- contact and collision information (if enabled)
- any custom sensor readings added for the SpiderBot task

Use the exact observation vector details in the environment class to keep this description synchronized with the code.

### Action Space

The action space typically includes:
- motor torque or velocity commands for the SpiderBot joints
- actuator setpoints for the simulated motors
- discrete or continuous control signals depending on the task configuration

The environment is designed to accept action vectors from an RL policy and map them to robot control commands inside the Isaac Sim task implementation.

### Reward Functions

The reward function is responsible for guiding learning toward the desired robot behavior. Common components include:
- `goal_reward`: reward for reaching or approaching the goal state
- `effort_penalty`: penalty for high joint torques or excessive control inputs
- `stability_penalty`: penalty for losing balance, flipping, or falling
- `contact_reward`: bonus for making or maintaining desired contact patterns
- `task_completion_bonus`: a larger reward when the task is completed successfully

Example reward structure:

```python
reward = 0.0
reward += goal_reward * goal_progress
reward -= effort_penalty * control_effort
reward -= stability_penalty * fall_detected
reward += task_completion_bonus if done_successfully else 0.0
```

Update this section with the exact reward terms used in your environment implementation.

## Installation

1. Install Isaac Lab by following the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
2. Clone or copy this repository outside of the Isaac Lab installation directory.
3. Install the project in editable mode using a Python interpreter that has Isaac Lab available:

```bash
python -m pip install -e source/SpiderBotTraining_1
```

## Usage

### List available tasks

```bash
python scripts/list_envs.py
```

### Run training

```bash
python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
```

### Run a zero-action sanity check

```bash
python scripts/zero_agent.py --task=<TASK_NAME>
```

### Run a random-action sanity check

```bash
python scripts/random_agent.py --task=<TASK_NAME>
```

## Set up IDE (Optional)

This repository includes a VSCode task for configuring the Python environment.

- Open the command palette: `Ctrl+Shift+P`
- Run `Tasks: Run Task`
- Select `setup_python_env`
- Provide the path to your Isaac Sim installation when prompted

If successful, a `.python.env` file will be created in `.vscode` containing the Python paths needed for source indexing.

## Extension Setup (Optional)

An example UI extension is available at `source/SpiderBotTraining_1/SpiderBotTraining_1/ui_extension_example.py`.

To enable it in Isaac Lab:
1. Add the absolute path to the repository `source` folder in Isaac Lab’s extension search paths.
2. Refresh the extension manager.
3. Enable the extension under the `Third Party` category.

## Results

Use this section to summarize training metrics and experiment outcomes.

- Training reward curves
- Episode success rates
- Average episode length
- Final evaluation performance
- Notes on improvement over baseline or previous runs

Example result summary:

- `Experiment 1`: reached stable walking behavior after 200k steps with an average episode reward of `X`
- `Experiment 2`: reduced control effort by `Y%` while maintaining task success
- `Experiment 3`: solved the navigation goal in `N` out of `M` evaluation episodes

## Code Formatting

Install pre-commit and run formatting checks:

```bash
pip install pre-commit
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

If VSCode does not index extension modules, add the repository path to `.vscode/settings.json` under `python.analysis.extraPaths`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/SpiderBotTraining_1"
    ]
}
```

### Pylance Crash

If Pylance crashes due to too many files being indexed, reduce the extra paths in `.vscode/settings.json` and exclude non-essential Omniverse packages.

Example exclusions:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"
"<path-to-isaac-sim>/extscache/omni.kit.*"
"<path-to-isaac-sim>/extscache/omni.graph.*"
"<path-to-isaac-sim>/extscache/omni.services.*"
```
