import numpy as np
import pandas as pd
from neural_network import Neural_Network
import utility as util

np.set_printoptions(threshold=np.inf)

df=pd.read_csv("./sample/train.csv", header=0)
nn = Neural_Network((df.shape[1]-1, 64, 64, 10), hidden_activation = "sigmoid", output_activation = "softmax", loss="categorical_crossentropy")

ans = 10
alpha = 0.01
batch = 7
epoch = int(df.shape[0]/batch)

def get_image(start, amount):
    img = []
    label = []
    for i in range(start, start+amount):
        temp = np.zeros(10)
        temp[df.iloc[i,0]] = 1.0
        label.append(temp)
        img.append(np.array(df.iloc[i,1:]))
    return (np.array(label), np.array(img))

def train():
    global batch, alpha
    count = 0
    for i in range(epoch):
        sample = get_image(i*batch, batch)
        label = sample[0]
        img = sample[1]
        output = nn.forward(img)
        nn.back_prop(label, img, batch, alpha)
        loss = nn.mse_loss(nn.forward(img), label)
        print(f"Epoch {i+1} Loss: {nn.mse_loss(nn.forward(img), label)}")
        print(output)
        count+=1
    return count

def save_model(num_epoch):
    epoch_directory = f"./model/epoch_{num_epoch}"
    util.create_directory(epoch_directory)
    weight = nn.getWeight()
    np.set_printoptions(threshold=np.inf)
    for i in range(len(weight)): np.savetxt(epoch_directory+f"/online_{i+1}.txt",weight[i], delimiter=" ", fmt="%s")
    print(f"Saved epoch_{num_epoch} successfully.")
    return

save_model(train())
