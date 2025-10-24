import abc
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from robot.sim import BimanualObs, BimanualSim
from train.train_utils import Job, Logs


class BimanualActor(nn.Module, abc.ABC):
  @abc.abstractmethod
  def forward(self, obs: BimanualObs) -> Tensor:
    raise NotImplementedError()


class Trainer(abc.ABC):
  @abc.abstractmethod
  def train(self, model: BimanualActor, num_epochs: int):
    raise NotImplementedError()


class BCTrainer(Trainer):
  def __init__(
    self,
    dataloader,
    checkpoint_frequency: int,
    job: Job
  ):
    self.dataloader = dataloader
    self.checkpoint_frequency = checkpoint_frequency
    self.job = job
  
  def train(self, model: BimanualActor, num_epochs: int):
    # TODO: move much of the details here into config
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
      for obs, action in self.dataloader:
        predicted_action = model(obs)
        loss = criterion(predicted_action, action)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
      if epoch % self.checkpoint_frequency == 0:
        self.job.save_checkpoint(model, 'bc-pretrain', epoch)


class RLTrainer(Trainer):
  def __init__(
    self,
    dataloader,
    checkpoint_frequency: int,
    job: Job,
    rollouts_per_epoch: int,
    rollout_step_count: int,
    init_sim: Callable[[], BimanualSim] = BimanualSim
  ):
    self.dataloader = dataloader
    self.checkpoint_frequency = checkpoint_frequency
    self.job = job
    self.rollouts_per_epoch = rollouts_per_epoch
    self.rollout_step_count = rollout_step_count
    self.init_sim = init_sim

  def train(self, model: BimanualActor, num_epochs: int):
    sim = self.init_sim()
    # TODO: implement RL. perhaps PPO?
    for epoch in range(num_epochs):
      for rollout_number in range(self.rollouts_per_epoch):
        obs = sim.get_obs()
        for rollout_step_number in range(self.rollout_step_count):
          predicted_action = model(obs)
          obs = sim.step(predicted_action)
          # TODO: RL reward collection / rollout buffer updates?
        # TODO: RL weight updates

      if epoch % self.checkpoint_frequency == 0:
        self.job.save_checkpoint(model, 'rl-train', epoch)

