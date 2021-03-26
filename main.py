import utils as nn
from paint import Paint

inn = 784
out = 10
dataset = 'train\\'
network = nn.Neural_network(inn, out, alpha=0.0001, weights='weights')
app = Paint(network)
app.Run()