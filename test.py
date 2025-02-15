import numpy as np
import pandas as pd
from neural_network import Neural_Network

df=pd.read_csv("./sample/train.csv", header=0)
nn = Neural_Network((df.shape[1]-1, 64, 64, 10))

def get_model(num_epoch):
    epoch_directory = f"./model/epoch_{num_epoch}"
    weight = [np.loadtxt(epoch_directory+f"/online_{i+1}.txt", delimiter=" ", dtype=float) for i in range(nn.layers-1)]
    nn.copy_network(weight)
    return

def get_image(start, amount):
    img = []
    label = []
    for i in range(start, start+amount):
        label.append(df.iloc[i,0])
        img.append(np.array(df.iloc[i,1:]))
    return (label, np.array(img))

def test(input, label):
    nn.forward(input)
    print("Prediction:", np.argmax(nn.forward(input)))
    print("Ans:", label)
    print(48*'-')
    return


sample = get_image(1000, 100)
get_model(6000)
for i in range(sample[1].shape[0]):
    test(sample[1][i], sample[0][i])

