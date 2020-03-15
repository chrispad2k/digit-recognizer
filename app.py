from PIL import Image
import numpy as np
import math

LOG = False

def getImage(file):
  if isinstance(file, str):
    image = Image.open(file, 'r')
    width, height = image.size
    pixel_values = list(image.getdata())

    if image.mode == 'L':
        channels = 1
    else:
        print("Image Mode must be black and white")
        return None

    data = np.array(pixel_values).reshape((-1, 1))

    return data

def squish(x, ACTUAL_BOUNDS=[0,255], DESIRED_BOUNDS=[0,1]):
  if not ACTUAL_BOUNDS[0] <= x <= ACTUAL_BOUNDS[1]:
    print("Number is not in the range {}".format(ACTUAL_BOUNDS))
    return None

  return  DESIRED_BOUNDS[0] + (x - ACTUAL_BOUNDS[0]) \
          * (DESIRED_BOUNDS[1] - DESIRED_BOUNDS[0]) \
          / (ACTUAL_BOUNDS[1] - ACTUAL_BOUNDS[0])

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Reading image and initializing network

file = 'data/training/0/img_1.jpg'
image_data = getImage(file)
squish_lambda = np.vectorize(squish)
a0 = squish_lambda(image_data)

size = [a0.size, 16, 16, 10]

if LOG:
  log = open("log/training.log","w+")
  log.write("Training Log\n")
  log.close()

for i in range(10):
  if LOG:
    log = open("log/training.log","a+")
    log.write("{}. Run:\n".format(i+1))

  print(a0)

  b1 = np.random.randint(-10, 10, size=(size[1], 1))
  w1 = np.random.uniform(low=-1, high=1, size=(size[1], size[0]))

  b2 = np.random.randint(-10, 10, size=(size[2], 1))
  w2 = np.random.uniform(low=-1, high=1, size=(size[2], size[1]))

  b3 = np.random.randint(-10, 10, size=(size[3], 1))
  w3 = np.random.uniform(low=-1, high=1, size=(size[3], size[2]))

  # Creating lambda for squishing

  a1_squish = np.vectorize(lambda x: squish(x, ACTUAL_BOUNDS=[-25,25], DESIRED_BOUNDS=[0, 1]))
  a2_squish = np.vectorize(lambda x: squish(x, ACTUAL_BOUNDS=[-25,25], DESIRED_BOUNDS=[0, 1]))
  a3_squish = np.vectorize(lambda x: squish(x, ACTUAL_BOUNDS=[-20,20], DESIRED_BOUNDS=[0, 1]))

  # Calculating Layers

  a1 = np.dot(w1, a0) + b1
  a1 = a1_squish(a1)

  a2 = np.dot(w2, a1) + b2
  a2 = a2_squish(a2)

  a3 = np.dot(w3, a2) + b3
  a3 = a3_squish(a3)

  if LOG:
    log.write("Input Layer\n")
    log.write(np.array_str(a0))
    log.write("\n")

    log.write("Layer 1\n")
    log.write("Activations\n")
    log.write(np.array_str(a1))
    log.write("\n")
    log.write("Weights\n")
    log.write(np.array_str(w1))
    log.write("\n")
    log.write("Biases\n")
    log.write(np.array_str(b1))
    log.write("\n")

    log.write("Layer 2\n")
    log.write("Activations\n")
    log.write(np.array_str(a2))
    log.write("\n")
    log.write("Weights\n")
    log.write(np.array_str(w2))
    log.write("\n")
    log.write("Biases\n")
    log.write(np.array_str(b2))
    log.write("\n")

    log.write("Output Layer\n")
    log.write("Activations\n")
    log.write(np.array_str(a3))
    log.write("\n")
    log.write("Weights\n")
    log.write(np.array_str(w3))
    log.write("\n")
    log.write("Biases\n")
    log.write(np.array_str(b3))
    log.write("\n")
