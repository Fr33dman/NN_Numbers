import utils as nn
from paint import Paint
from PIL import Image
import numpy as np
import os

inn = 784
out = 10
dataset = 'train\\'
network = nn.Neural_network(inn, out, alpha=0.00000001)
for i in range(100):
    for item in os.listdir(dataset):
        number = int(item.split('.')[0][-1])
        right_answer = np.array(np.zeros((1,10)))
        right_answer[0][number] = 100
        with Image.open(dataset+item) as photo:
            img = np.array(photo)#/255
            img = np.matrix(img.ravel())
        network.Learning(img, right_answer[0])
        try:
            with open('cache.txt', 'a') as cache:
                for error in network.error:
                    cache.write(str(error) + '\t')
                cache.write('\n')
        except PermissionError:
            print('PermissionError')
network.save_weights()

app = Paint(network)
app.Run()