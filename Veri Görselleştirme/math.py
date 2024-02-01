import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

sutun1 = abs(np.random.normal(1, 12, 100))
sutun2 = abs(np.random.normal(2, 8, 100))
sutun3 = abs(np.random.normal(3, 2, 100))
sutun4 = abs(np.random.normal(10000, 1500000, 100))

x = np.c_[sutun1, sutun2, sutun3, sutun4]
y = [(np.random.randint(0,4)) for i in range(100)]

data = pd.DataFrame()

data['col1'] = sutun1
data['col2'] = sutun2
data['col3'] = sutun3
data['col4'] = sutun4

plt.subplot(2,2,1)
plt.title('col1')
plt.scatter(y,sutun1, color='green',label = 'col1')

plt.subplot(2,2,2)
plt.title('col2')
plt.scatter(y,sutun1, color='blue',label = 'col2')

plt.subplot(2,2,3)
plt.title('col3')
plt.scatter(y,sutun1, color='orange',label = 'col3')

plt.subplot(2,2,4)
plt.title('col4')
plt.scatter(y,sutun1, color='magenta',label = 'col4')

plt.savefig('Veri Görselleştirme.jpg')

plt.show()