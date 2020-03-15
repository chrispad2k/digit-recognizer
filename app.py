import numpy as np
from network import Network
from utils import getImage, squish

# Reading image and initializing network

file = 'data/training/0/img_1.jpg'
image_data = getImage(file)
squish_lambda = np.vectorize(squish)
data = squish_lambda(image_data)

shape = data.size, 16, 16, 10

network = Network(shape)
network.create()

output = network.run(data)

print(output)
