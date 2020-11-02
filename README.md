# SimPLe PyTorch

Unofficial PyTorch implementation of the [SimPLe algorithm](https://arxiv.org/abs/1903.00374) for the Arcade Learning Environment's Atari 2600 games.

- [Installation](#installation)
- [How to use](#how-to-use)
- [Stats](#stats)

**TODO** gif.

## Installation

This program requires **python 3.7**.

### Using CUDA

This program uses CUDA 10.2 by default. The following command should install it by default.

**TODO** check CUDA for tensorflow-gpu

Run the following command to install the dependencies:
```bash
pip install stable-baselines==2.10.1 torch==1.6.0 tensorflow-gpu==1.14.0 tqdm==4.49.0 numpy==1.16.4
```

### Without CUDA

Run the following command to install the dependencies:
```bash
pip install stable-baselines==2.10.1 torch==1.6.0 tensorflow==1.14.0 tqdm==4.49.0 numpy==1.16.4
```

### Install wandb (optional)

You can use [wandb](https://www.wandb.com/) to track your experiment:
```bash
pip install wandb==0.10.8
```

To use wandb, pass the flag `--use-wandb` when running the program. See [How to use](#how-to-use) for more details about flags.

## How to use

CUDA is enabled by default, see the following section to disable it.

To run the program, run the following command from the `src` directory:
```bash
python simple.py
```

### Disable CUDA

To disable CUDA, pass the flag `--device cpu` to the command line. See the next section for more information about flags.

### Flags

You can pass multiple flags to the command line, a summary is printed at launch time.
The most useful flags are described in the following table:

| Flag | Value | Default | Description |
| ---- | ----- | ------- | ----------- |
| --agents | Any positive integer | 4 | The number of parallel environments to train the PPO agent on |
| --device | Any string accepted by [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#device-doc) | cuda | Sets the PyTorch's device |
| --env-name | Any string accepted by [gym.make](https://gym.openai.com/docs/#environments) | FreewayDeterministic-v4 | Sets the gym environment | 

The following boolean flags are set to `False` if not passed to the command line:

| Flag | Description |
| ---- | ----------- |
| --load-models | Loads models from `src/models/` and bypasses training |
| --render-training | Renders the environments during training |
| --save-models | Save the models after each epoch |
| --use-wandb | Enables [wandb](https://www.wandb.com/) to track the experiment |

For example, to run the program without CUDA and to render the environments during training, run:
```bash
python simple.py --device cpu --render-training
```


## Stats

- Using CUDA, training the agent and the models takes ~16h on a Nvidia GTX 1070 GPU
- Training the models requires 8Gb of GPU VRAM
- Training the PPO agents on 4 agents requires 16Gb of RAM. If your machine runs out of memory, consider reducing the number of agents by passing the `--agents` flag 
- This program was tested on Ubuntu 20.04.1 LTS