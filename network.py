import numpy as np
from utils import squish

class Network:
  def __init__(self):
    self.__init__((2, 16, 16, 1))

  def __init__(self, shape):
    # Error Checking
    if len(shape) < 3:
      raise ValueError('A neural network must have at least 3 layers.')
    for l in shape:
      if l <= 0:
        raise ValueError('A layer must have at least 1 neuron.')

    self.shape = shape
    self.size = len(shape)

  def __del__(self):
      self.shape = None
      self.size = None
      self.layers = None

  def create(self):
    self.layers = []

    for i in range(self.size):
      biases = None
      weights = None

      if i != 0:
        biases = np.random.randint(-10, 10, size=(self.shape[i], 1))
        weights = np.random.uniform(low=-1, high=1, size=(self.shape[i], self.shape[i-1]))

      self.layers.append({
        'id': i,
        'activations': None,
        'weights': weights,
        'biases': biases
      })

  def run(self, input):
    if not isinstance(input, np.ndarray):
      raise ValueError('The input must be of the type <numpy.ndarray>')
    input_shape = self.shape[0], 1
    if input.shape != input_shape:
      raise ValueError('The input array must be of the shape {}'.format(input_shape))

    self.layers[0]['activations'] = input

    for i in range(self.size-1):
      prev_activations = self.layers[i]['activations']
      layer = self.layers[i+1]
      squish_bounds = self.shape[i+1] + np.amax(layer['biases'])
      squish_lambda = np.vectorize(lambda x: squish(x, ACTUAL_BOUNDS=[-squish_bounds,squish_bounds], DESIRED_BOUNDS=[0, 1]))

      activations = np.dot(layer['weights'], prev_activations) + layer['biases']
      activations = squish_lambda(activations)

      layer['activations'] = activations

    return self.layers[self.size-1]['activations']
