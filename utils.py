from PIL import Image
import numpy as np
import math

def getImage(file):
  if isinstance(file, str):
    image = Image.open(file, 'r')
    width, height = image.size
    pixel_values = list(image.getdata())

    if image.mode == 'L':
        channels = 1
    else:
        print('Image Mode must be black and white')
        return None

    data = np.array(pixel_values).reshape((-1, 1))

    return data

def squish(x, ACTUAL_BOUNDS=[0,255], DESIRED_BOUNDS=[0,1]):
  if not ACTUAL_BOUNDS[0] <= x <= ACTUAL_BOUNDS[1]:
    print('Number is not in the range {}'.format(ACTUAL_BOUNDS))
    return None

  return  DESIRED_BOUNDS[0] + (x - ACTUAL_BOUNDS[0]) \
          * (DESIRED_BOUNDS[1] - DESIRED_BOUNDS[0]) \
          / (ACTUAL_BOUNDS[1] - ACTUAL_BOUNDS[0])

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

