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
    obs = None
    while len(scores) < episodes:
        if dones.all():
            if env is not None:
                env.close()
            env = make_envs(env_name, agents, device, **kwargs)
            dones = np.array([False] * agents)
            obs = env.reset()

        with torch.no_grad():
            action = agent.act(obs)[1]

        obs, reward, done, infos = env.step(action)

        for i in range(agents):
            if 'r' in infos[i].keys() and not dones[i]:
                assert done[i] == True
                dones[i] = True
                scores.append(infos[i]['r'])
                if verbose:
                    t.set_postfix({'mean_score': np.mean(scores), 'std_score': np.std(scores)})
                    t.update()

    if env is not None:
        env.close()

    return scores
