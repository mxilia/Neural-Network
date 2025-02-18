import numpy as np
import pandas as pd
from neural_network import Neural_Network

cnt = 0
df=pd.read_csv("./sample/train.csv", header=0)
nn = Neural_Network((df.shape[1]-1, 64, 64, 10), hidden_activation = "sigmoid", output_activation = "softmax", loss="categorical_crossentropy")

def get_image(start, amount):
    img = []
    label = []
    hot_label = []
    for i in range(start, start+amount):
        temp = np.zeros(10)
        temp[df.iloc[i,0]] = 1.0
        hot_label.append(temp)
        label.append(df.iloc[i,0])
        img.append(np.array(df.iloc[i,1:]/255))
    return (label, np.array(img), np.array(hot_label))

def test(input, label):
    output = nn.forward(input)
    # print(output, np.sum(output))
    print("Prediction:", np.argmax(output))
    print("Ans:", label)
    global cnt
    if(np.argmax(output)==label): cnt+=1
    print(48*'-')
    return

sample = get_image(round(df.shape[0]/2), df.shape[0]-(round(df.shape[0]/2)))
nn.load_model(f"epoch_{700}", "w")
for i in range(sample[1].shape[0]):
    test(np.array([sample[1][i]]), sample[0][i])
loss = nn.loss_func(nn.forward(sample[1]), sample[2])
acc = cnt/sample[1].shape[0]*100
print("Loss:", loss)
print("Acc:", acc)

