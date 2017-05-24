import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('temp_prediction_vs_observations.csv')

a = df['id'].tolist()
b = df['prediction'].tolist()
c = df['ground_truth'].tolist()

a = [0.55, 0.575, 0.6, 0.625, 0.7, 0.725]
b = [1.917645758, 1.9297372098, 1.8849759864, 1.882434859, 1.8809178383,1.8967282]
c = [1.897, 1.891, 1.896, 1.886, 1.888, 1.895]

fig, ax = plt.subplots()
ax.plot(a,b, '-',linewidth=2, label='predictions')
ax.plot(a,c, ':',linewidth=2, label='observations')
# legend = ax.legend(loc='top right', shadow=True)
legend = ax.legend(shadow=True)

plt.title('prediction vs observation')
plt.xlabel('id')
plt.ylabel('value')

plt.show()
