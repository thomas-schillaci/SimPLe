import torch

from a2c_ppo_acktr import ppo
from a2c_ppo_acktr.policy import Policy
from a2c_ppo_acktr.rollout_storage import RolloutStorage


class PPO:

    def __init__(self,
                 env,
                 config,
                 num_steps=128,
                 lr=2.5e-4,
                 clip_param=0.1,
                 value_loss_coef=0.5,
                 num_mini_batch=4,
                 entropy_coef=0.01,
                 eps=1e-5,
                 max_grad_norm=0.5,
                 ppo_epoch=4
                 ):

        self.env = env
        self.config = config
        self.lr = lr
        self.num_steps = num_steps
        self.num_processes = self.env.num_envs

        self.actor_critic = Policy(
            self.env.observation_space.shape,
            self.env.action_space,
            base_kwargs={'recurrent': False}
        )
        self.actor_critic.to(config.device)

        self.agent = ppo.PPO(
            self.actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm
        )

        self.eval_recurrent_hidden_states = None

    def learn(self, steps):
        num_updates = steps // self.num_steps // self.num_processes

        rollouts = RolloutStorage(
            self.num_steps,
            self.num_processes,
            self.env.observation_space.shape,
            self.env.action_space,
            self.actor_critic.recurrent_hidden_state_size
        )

        obs = self.env.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(self.config.device)

        for j in range(num_updates):
            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step]
                    )

                # Obser reward and next obs
                obs, reward, done, infos = self.env.step(action.detach().cpu())

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                rollouts.insert(
                    obs,
                    recurrent_hidden_states,
                    action,
                    action_log_prob,
                    value,
                    reward,
                    masks,
                    bad_masks
                )

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]
                ).detach()

            rollouts.compute_returns(next_value, True, self.config.ppo_gamma, 0.95, False)
            value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
            rollouts.after_update()

            losses = {'ppo_value_loss': float(value_loss), 'ppo_action_loss': float(action_loss)}
            if self.config.use_wandb:
                import wandb
                wandb.log(losses)

        return losses

    def set_env(self, env):
        self.env = env
        self.num_processes = env.num_envs

    def save(self, path):
        torch.save(self.actor_critic, path)

    def load(self, path):
        self.actor_critic = torch.load(path)

    def init_eval(self):
        self.eval_recurrent_hidden_states = torch.zeros(
            self.num_processes,
            self.actor_critic.recurrent_hidden_state_size,
            device=self.config.device
        )

    def predict(self, obs):
        assert self.eval_recurrent_hidden_states is not None
        with torch.no_grad():
            _, action, _, self.eval_recurrent_hidden_states = self.actor_critic.act(
                obs,
                self.eval_recurrent_hidden_states,
                torch.zeros(self.num_processes, 1, device=self.config.device),
                deterministic=True
            )

        return action
