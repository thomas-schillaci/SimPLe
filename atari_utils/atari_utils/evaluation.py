import torch
import numpy as np
from tqdm import trange

from atari_utils.envs import make_envs


def evaluate(agent, env_name, device, agents=1, episodes=30, verbose=True, **kwargs):
    scores = []
    if verbose:
        t = trange(episodes, desc='Evaluating agent')

    env = None
    dones = np.array([True])
    steps = [0] * agents
    metrics = {}
    obs = None
    while len(scores) < episodes:
        if dones.all():
            if env is not None:
                env.close()
            env = make_envs(env_name, agents, device, **kwargs)
            dones = np.array([False] * agents)
            steps = [0] * agents
            obs = env.reset()

        with torch.no_grad():
            action = agent.act(obs)[1]

        obs, reward, done, infos = env.step(action)

        for i in range(agents):
            if not dones[i]:
                steps[i] += 1
            if 'r' in infos[i].keys() and not dones[i]:
                assert done[i]
                dones[i] = True
                scores.append(infos[i]['r'])
                metrics.update({
                        'eval_score_mean': np.mean(scores),
                        'eval_score_std': np.std(scores),
                        'eval_steps_mean': np.mean(steps),
                        'eval_steps_std': np.std(steps)
                    })
                if verbose:
                    t.set_postfix(metrics)
                    t.update()

    if env is not None:
        env.close()

    return metrics
