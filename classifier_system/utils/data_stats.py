import matplotlib.pyplot as plt
import pandas as pd

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Unrelated', 'Pro-ED', 'Pro-recovery'
df = pd.read_csv('../data/dataset_training_balanced.csv')
proed = df['label'].apply(lambda label: True if label == 'proED' else False)
a = len(proed[proed == True].index)
prorec = df['label'].apply(lambda label: True if label == 'prorecovery' else False)
b = len(prorec[prorec == True].index)
unrelated = df['label'].apply(lambda label: True if label == 'unrelated' else False)
c = len(unrelated[unrelated == True].index)
sizes = [c, a, b]
print(c, a, b, a+b+c)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig('label-distribution-dataset-bal.png')
plt.show()
