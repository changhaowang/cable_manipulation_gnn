"""
Script for checking the data saved in the tfrecord.
"""

import numpy as np
import collections
import functools
import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)
from learning_to_simulate import reading_utils

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.

data_path = 'learning_to_simulate/datasets/Rope'

def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())

def prepare_inputs(tensor_dict):

  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])

  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]

  # Remove the target from the input.
  tensor_dict['position'] = pos[:, :-1]

  # Similarly, we change the shape of the velocity (the control input from the robot on each node)
  vel = tensor_dict['velocity']
  vel = tf.transpose(vel, perm=[1, 0, 2])

  tensor_dict['velocity'] = vel[:, :-1]

  # Compute the number of particles per example.
  num_particles = tf.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
  # # change the shape of the particle type
  # if len(tensor_dict['particle_type']) != num_particles:
  #   print('Change shape of the particle_type')
  #   tensor_dict['particle_type'] = [5] * num_particles.numpy()

  if 'step_context' in tensor_dict:
    # Take the input global context. We have a stack of global contexts,
    # and we take the penultimate since the final is the target.
    tensor_dict['step_context'] = tensor_dict['step_context'][-2]
    # Add an extra dimension for stacking via concat.
    tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
  return tensor_dict, target_position


metadata = _read_metadata(data_path)
# Create a tf.data.Dataset from the TFRecord.
ds = tf.data.TFRecordDataset([os.path.join(data_path, 'train.tfrecord')])
ds = ds.map(functools.partial(
    reading_utils.parse_serialized_simulation_example, metadata=metadata))

# Splits an entire trajectory into chunks of 7 steps.
# Previous 5 velocities, current velocity and target.
split_with_window = functools.partial(
    reading_utils.split_trajectory,
    window_length=INPUT_SEQUENCE_LENGTH + 1)
ds = ds.flat_map(split_with_window)
# Splits a chunk into input steps and target steps
ds = ds.map(prepare_inputs)

# Test 2
counter = 0
for ele in ds:
    counter +=1
    print(counter)

ds = ds.repeat()
ds = ds.shuffle(512)
