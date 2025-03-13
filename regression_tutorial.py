
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.datasets import fetch_openml

boston = fetch_openml(name='boston')

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["TARGET"] = boston.target
df.head()

sns.pairplot(df)
plt.show()

cols = ["TARGET", "INDUS", "RM", "LSTAT", "NOX", "DIS"]
sns.pairplot(df[cols])
plt.show()

data = torch.from_numpy(df[cols].values).float()
data.shape

x = data[:, 1:]
y = data[:, :1]
x.shape, y.shape

epochs = 2000
lr = 1e-3
print_interval = 100

model = nn.Linear(x.size(-1), y.size(-1))
model

optimizer = optim.SGD(model.parameters(), lr = lr)

for i in range(epochs):
  y_pred = model(x)
  loss = F.mse_loss(y, y_pred)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (i + 1) % print_interval == 0:
    print("Epoch %d: loss=%4e" %(i+1, loss))

df = pd.DataFrame(torch.cat([y, y_pred], dim = 1).detach_().numpy(), columns = ["y", "y_pred"])
sns.pairplot(df, height = 5)
plt.show()