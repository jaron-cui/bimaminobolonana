from dataclasses import dataclass
from datetime import datetime
import glob
import os
from typing import List, cast
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class Job:
  """
  Represents a training job and provides a clean, programmatic way of interacting with output files.

  root_output_folder/
    date_tag/
      config.yaml
      debug.log
      checkpoint/
        <training-stage-name1>/
          0.pt
          ...
          n.pt
        <training-stage-name2>/
          ...
      artifact/
        ...
  """
  path: Path
  date: datetime
  tag: str

  @property
  def config_path(self) -> Path:
    return self.path / 'config.yaml'

  @property
  def debug_log_path(self) -> Path:
    return self.path / 'debug.log'

  def load_checkpoint(self, training_stage: str, epoch: int) -> nn.Module:
    return torch.load(self._checkpoint_path(training_stage, epoch))

  def save_checkpoint(self, model: nn.Module, training_stage: str, epoch: int) -> Path:
    checkpoint_path = self._checkpoint_path(training_stage, epoch)
    os.makedirs(checkpoint_path.parent, exist_ok=True)
    # torch.save(model, checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path

  def _checkpoint_path(self, training_stage: str, epoch: int) -> Path:
    return self.path / f'checkpoint/{training_stage}/{epoch}.pt'

  def __str__(self) -> str:
    return f'Job({self.path.name})'


class Logs:
  DATE_FORMAT = '%m%d%y-%H%M%S'

  def __init__(self, path: str):
    self.root_dir = Path(path)
    if not self.root_dir.exists():
      raise ValueError(f'Path {path} does not exist. Provide the absolute path to your training output directory.')

  def jobs(self) -> List[Job]:
    jobs = []
    for path in glob.glob('*', root_dir=self.root_dir):
      job = Logs._parse_job_path(Path(path), silent_error=True)
      if job is None:
        continue
      jobs.append(job)
    return jobs

  def create_new_job(self, tag: str) -> Job:
    if '_' in tag:
      raise ValueError(f'Job tag `{tag}` cannot contain underscores. Remove them or replace them with hyphens.')
    date = datetime.now()
    path = self.root_dir / f'{date.strftime(Logs.DATE_FORMAT)}_{tag}'
    if path.exists():
      raise ValueError(f'Job at {path} already exists!')
    path.mkdir()
    return cast(Job, self._parse_job_path(path))

  @staticmethod
  def _parse_job_path(job_path: Path, silent_error: bool = False):
    parts = job_path.name.split('_')
    parsers = [lambda d: datetime.strptime(d, Logs.DATE_FORMAT), lambda t: t]
    if len(parts) < len(parsers):
      if silent_error:
        return None
      raise ValueError('Invalid job name.')
    parsed_parts = [parse(part) for part, parse in zip(parts, parsers)]
    return Job(
      path=job_path,
      date=parsed_parts[0],
      tag=parsed_parts[1]
    )
