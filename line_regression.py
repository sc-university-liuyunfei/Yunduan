import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

#超参数设定
input_size = 1
output_size = 1
learning_rate = 0.001

#构造数据
xtrain = np.array([[2.3],[4.4],[3.7],[6.1],[7.3],[2.1],[5.6],[7.7],[8.7],[4.1],[6.7],[6.1],[7.5],[2.1],[7.2],[5.6],[5.7],[7.7],[3.1]],dtype=np.float32)
ytrain = np.array([[3.7],[4.76],[4.],[7.1],[8.6],[3.5],[5.4],[7.6],[7.9],[5.3],[7.4],[7.5],[8.5],[3.2],[8.7],[6.4],[6.6],[7.9],[5.3]],dtype=np.float32)

#绘制散点图
plt.ion()
plt.figure()
plt.scatter(xtrain,ytrain)
plt.xlabel('xtrain 数据')
plt.ylabel('ytrain 数据')
plt.pause(5)
plt.show()
plt.savefig('linearscatter.png')
plt.close()

#建立模型
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self, x):
        out = self.linear(x)
        return out
model = LinearRegression(input_size,output_size)

#定义优化器
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#定义损失函数
criterion = nn.MSELoss()

#开始训练
num_epochs = 1000
for epoch in range(num_epochs):
    #numpy -> tensor -> 变量
    inputs = Variable(torch.from_numpy(xtrain))
    targets = Variable(torch.from_numpy(ytrain))

    outputs = model(inputs)
    loss = criterion(outputs,targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #显示数据
    if (epoch+1) % 50 ==0:
        print('Epoch [%d/%d],Loss: %.4f' % (epoch+1,num_epochs,loss.item()))

#绘制直线
#训练模型完成后，使用model.train()和model.eval(),固定模型，使其参数在测试时保持不变
model.eval()

#将模型中某个变量提出了进行数据转换
predicted = model(Variable(torch.from_numpy(xtrain))).data.numpy()

plt.plot(xtrain,ytrain,'ro')#plt.plot()是可以画散点的，取决于第三个参数，如果不指定，则默认为依次直线连接，如果特别指定，如'o'就表示每个点是圆圈
plt.plot(xtrain,predicted,label='predict')
plt.legend(loc='best')
plt.pause(4)
plt.show()
plt.savefig('linearRegression.png')