from abc import abstractmethod


class Logger:
    @abstractmethod
    def log(self, x):
        pass


class WandBLogger(Logger):

    def log(self, x):
        import wandb
        wandb.log(x)
