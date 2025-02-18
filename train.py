import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_network import Neural_Network

np.set_printoptions(threshold=np.inf)

df=pd.read_csv("./sample/train.csv", header=0)
nn = Neural_Network((df.shape[1]-1, 64, 64, 10), hidden_activation = "sigmoid", output_activation = "softmax", loss="categorical_crossentropy")

list_loss = []
list_acc = []

ans = 10
alpha = 0.65
batch = 30
sample_size = df.shape[0]/2
epoch = int(sample_size/batch)
time = 1

def get_image(start, amount):
    img = []
    label = []
    for i in range(start, start+amount):
        temp = np.zeros(10)
        temp[df.iloc[i,0]] = 1.0
        label.append(temp)
        img.append(np.array(df.iloc[i,1:]/255.0))
    return (np.array(label), np.array(img))

def train():
    global batch, alpha
    count = 0
    for t in range(time):
        for i in range(epoch):
            sample = get_image(i*batch, batch)
            label = sample[0]
            img = sample[1]
            nn.forward(img)
            nn.back_prop(label, img, alpha)
            list_loss.append(nn.loss_func(nn.forward(img), label))
            count+=1
    return count

def plot():
    plt.figure(figsize=(6,4), dpi=100)
    plt.plot(list_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    return

nn.save_model(f"epoch_{train()}", "w")
plot()
