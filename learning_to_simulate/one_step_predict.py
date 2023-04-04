# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=line-too-long
"""Training script for https://arxiv.org/pdf/2002.09405.pdf.

Example usage (from parent directory):
`python -m learning_to_simulate.train --data_path={DATA_PATH} --model_path={MODEL_PATH}`

Evaluate model from checkpoint (from parent directory):
`python -m learning_to_simulate.train --data_path={DATA_PATH} --model_path={MODEL_PATH} --mode=eval`

Produce rollouts (from parent directory):
`python -m learning_to_simulate.train --data_path={DATA_PATH} --model_path={MODEL_PATH} --output_path={OUTPUT_PATH} --mode=eval_rollout`
"""
# pylint: enable=line-too-long
import collections
import functools
import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging
from queue import Queue
import numpy as np
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import tree

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)

from learning_to_simulate import learned_simulator
from learning_to_simulate import noise_utils
from learning_to_simulate import reading_utils

default_dataset_dir = 'learning_to_simulate/datasets/Rope'
default_model_dir = 'learning_to_simulate/models/Rope'
default_output_dir = 'learning_to_simulate/rollouts/Rope'

flags.DEFINE_enum(
    'mode', 'eval_rollout', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'train', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', default_dataset_dir, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 20, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', default_model_dir,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', default_output_dir,
                    help='The path for saving outputs (e.g. rollouts).')


FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 5  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3 # default 3
DIM = 2



def _combine_std(std_x, std_y):
  return np.sqrt(std_x**2 + std_y**2)


def _get_simulator(model_kwargs, metadata, acc_noise_std, vel_noise_std):
  """Instantiates the simulator."""
  # Cast statistics to numpy so they are arrays when entering the model.
  cast = lambda v: np.array(v, dtype=np.float32)
  acceleration_stats = Stats(
      cast(metadata['acc_mean']),
      _combine_std(cast(metadata['acc_std']), acc_noise_std))
  velocity_stats = Stats(
      cast(metadata['vel_mean']),
      _combine_std(cast(metadata['vel_std']), vel_noise_std))
  normalization_stats = {'acceleration': acceleration_stats,
                         'velocity': velocity_stats}
  if 'context_mean' in metadata:
    context_stats = Stats(
        cast(metadata['context_mean']), cast(metadata['context_std']))
    normalization_stats['context'] = context_stats

  simulator = learned_simulator.LearnedSimulator(
      num_dimensions=metadata['dim'],
      connectivity_radius=metadata['default_connectivity_radius'],
      graph_network_kwargs=model_kwargs,
      boundaries=metadata['bounds'],
      num_particle_types=NUM_PARTICLE_TYPES,
      normalization_stats=normalization_stats,
      particle_type_embedding_size=16)
  return simulator


def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())


def one_step(data_path,
            noise_std,
            current_positions,
            current_controls,
            latent_size=128,
            hidden_size=128,
            hidden_layers=2,
            message_passing_steps=10):
    metadata = _read_metadata(data_path)
    model_kwargs = dict(
      latent_size=latent_size,
      mlp_hidden_size=hidden_size,
      mlp_num_hidden_layers=hidden_layers,
      num_message_passing_steps=message_passing_steps)
    simulator = _get_simulator(model_kwargs, metadata,
                               acc_noise_std=noise_std,
                               vel_noise_std=noise_std)
    next_position = simulator(
        current_positions,
        current_controls,
        n_particles_per_example=tf.constant(13,dtype=tf.int64),
        particle_types=tf.constant([5,5,5,5,5,5,5,5,5,5,5,5,5],dtype=tf.int64),
        global_context=None)
    return next_position

def get_one_step_input():
    """Gets initial_control and initial_position for one step predict  from pybullet."""
    #temporarily, load data from  .txt files 
    f = open('learning_to_simulate/data_pos1.txt', 'r')
    txt = f.read()
    sequence_pos=Queue(maxsize=INPUT_SEQUENCE_LENGTH)
    for data in txt.split('\n\n'):
        pos=[]
        for line in data.split('\n'):
            node_pos=[]
            for i in line.replace('[', '').replace(']', '').replace('\n', '' ).split(' '):
                if i is not ' ' and i is not '':
                        node_pos.append(float(i))
                        if len(node_pos)==2:
                            pos.append(np.array(node_pos))
        if sequence_pos.full():
            sequence_pos.get()
            sequence_pos.put(np.array(pos))
        else:
            sequence_pos.put(np.array(pos))
    f.close()
    f = open("learning_to_simulate/data_vel1.txt")
    txt = f.read()
    sequence_control=Queue(maxsize=INPUT_SEQUENCE_LENGTH)
    for data in txt.split('\n\n'):
        control=[]
        for line in data.split('\n'):
            node_control=[]
            for i in line.replace('[', '').replace(']', '').replace('\n', '' ).split(' '):
                if i is not ' ' and i is not '':
                    node_control.append(float(i))
                    if len(node_control)==2:
                        control.append(np.array(node_control))
        if sequence_control.full():
            sequence_control.get()
            sequence_control.put(np.array(control))
        else:
            sequence_control.put(np.array(control))
    f.close()
    rope_control_stack = []
    while  not sequence_control.empty():
        rope_control_stack .append(sequence_control.get())
    rope_pose_stack = []
    while  not sequence_pos.empty():
        rope_pose_stack .append(sequence_pos.get())
    rope_pose_stack = np.array(rope_pose_stack)
    rope_control_stack = np.array(rope_control_stack)
    current_positions =  np.transpose(rope_pose_stack, [1, 0, 2])
    current_controls = np.transpose(rope_control_stack, [1, 0, 2])
    return current_controls, current_positions

def one_step_prediction(simulator, features):
  """Rolls out a trajectory by applying the model in sequence."""
  start_index = 0
  initial_positions = features['position'][:, start_index:start_index+INPUT_SEQUENCE_LENGTH]
  initial_control = features['velocity'][:, start_index:start_index+INPUT_SEQUENCE_LENGTH]
  
  next_position = simulator(
        initial_positions,
        initial_control,
        n_particles_per_example=tf.convert_to_tensor(np.array([13])), #features['n_particles_per_example']
        particle_types=tf.convert_to_tensor((5*np.ones(13,)).astype(int)),#features['particle_type'],
        global_context=None)

  predictions = tf.TensorArray(size=1, dtype=tf.float32)
  predictions = predictions.write(0, next_position)

  output_dict = {
      'predicted_rollout': predictions.stack(),
  }
  return output_dict

def get_one_step_estimator_fn(data_path,
                             noise_std,
                             latent_size=128,
                             hidden_size=128,
                             hidden_layers=2,
                             message_passing_steps=10):
  """Gets the model function for tf.estimator.Estimator."""
  metadata = _read_metadata(data_path)

  model_kwargs = dict(
      latent_size=latent_size,
      mlp_hidden_size=hidden_size,
      mlp_num_hidden_layers=hidden_layers,
      num_message_passing_steps=message_passing_steps)

  def estimator_fn(features, labels,  mode):
    del labels
    simulator = _get_simulator(model_kwargs, metadata,
                               acc_noise_std=noise_std,
                               vel_noise_std=noise_std)
    rollout_op = one_step_prediction(simulator, features)
    rollout_op = tree.map_structure(lambda x: x[tf.newaxis], rollout_op)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=rollout_op)

  return estimator_fn


if __name__ == '__main__':
  tf.disable_v2_behavior()
  FLAGS(sys.argv)
  current_controls,current_positions=get_one_step_input()
  input_dict = {}
  input_dict['position'] = current_positions.astype(np.float32)
  input_dict['velocity'] = current_controls.astype(np.float32)
  rollout_estimator = tf.estimator.Estimator(get_one_step_estimator_fn(FLAGS.data_path, FLAGS.noise_std), model_dir=FLAGS.model_path)
  my_input_fn = tf.estimator.inputs.numpy_input_fn(x=(input_dict), shuffle=False, batch_size=13, num_epochs=1)
  rollout_iterator = rollout_estimator.predict(input_fn = my_input_fn)

  for prediction in rollout_iterator:
      print(prediction)