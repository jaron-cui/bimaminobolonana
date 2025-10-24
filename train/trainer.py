import abc
from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

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
    job: Job,
    on_log_message: Callable[[str], None] = print  # TODO: not sure if I like this on_log_message thing I wrote for now
  ):
    self.dataloader = dataloader
    self.checkpoint_frequency = checkpoint_frequency
    self.job = job
    self.log_message = on_log_message
  
  def train(self, model: BimanualActor, num_epochs: int):
    # TODO: make the optimizer, criterion, etc... passed in via constructor args so we
    #       can use hydra to instantiate BCTrainer with different settings
    # TODO: route logs into self.job.debug_log_path file
    self.log_message(f'Training model for {num_epochs} epochs.')
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
      epoch_loss = 0
      for obs, action in tqdm(self.dataloader, desc=f'Epoch {epoch}'):
        predicted_action = model(obs)
        # TODO: we'll want the (obs, action) from dataloader to use the tensor versions of dataclass defined
        #       at the top of train/dataset.py instead of the robot/sim.py numpy versions
        #       so that training is faster
        loss = criterion(predicted_action, torch.from_numpy(action.array))
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
      self.log_message(f' - Epoch {epoch} loss: {epoch_loss:.4f}')
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

