import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot

data_s_path = '../data/semi_auto_shuffled.csv'
data_test_path = '../data/dataset_reddit.csv'

df_test = pd.read_csv(data_test_path)
y_true = df_test['label'].values.tolist()
post = df_test['text'].values.tolist()
labels_dict = {'unrelated': 0, 'proED': 1, 'prorecovery': 2}
for i in range(len(y_true)):
    y_true[i] = labels_dict[y_true[i]]

preds_ernie = np.load('../preds/preds_test_ernie_semi.npy')
preds_ernie = np.argmax(preds_ernie, axis=-1).tolist()
print(preds_ernie, len(preds_ernie))
print(y_true, len(y_true))

indexes = []
for i in range(len(preds_ernie)):
    if preds_ernie[i] == 1:
        print('Prediction:', preds_ernie[i])
        print('Post:', post[i])
        print('True label:', y_true[i], '\n')
    if y_true[i] == 1 and preds_ernie[i] != 1:
        print('Missed prediction:', preds_ernie[i])
        print('Post:', post[i])
        print('True label:', y_true[i], '\n')

# conf_mat = confusion_matrix(y_true, preds_ernie)
# cmd = ConfusionMatrixDisplay(conf_mat, display_labels=['Unrelated', 'Pro-ED', 'Pro-recovery'])
# cmd.plot(cmap='Blues')
# pyplot.savefig('cm_ernie_reddit_semi.png')
