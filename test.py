import numpy as np
import pandas as pd
from neural_network import Neural_Network

df=pd.read_csv("./sample/train.csv", header=0)
nn = Neural_Network((df.shape[1]-1, 64, 64, 10), hidden_activation = "sigmoid", output_activation = "sigmoid", loss="mean_squared_error")

def get_image(start, amount):
    img = []
    label = []
    for i in range(start, start+amount):
        label.append(df.iloc[i,0])
        img.append(np.array(df.iloc[i,1:]))
    return (label, np.array(img))

def test(input, label):
    output = nn.forward(input)
    print(output, np.sum(output))
    print("Prediction:", np.argmax(output))
    print("Ans:", label)
    print(48*'-')
    return

sample = get_image(10000, 100)
nn.load_model(f"epoch_{2100}", "w")
for i in range(sample[1].shape[0]):
    test(sample[1][i], sample[0][i])

