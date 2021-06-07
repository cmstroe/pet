from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd 

k_fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
data =  pd.read_csv("all_causes.csv", names = ['label', 'text'])
X = data.text
y = data.label

train, test = [], []
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
     train.append(pd.DataFrame(columns = ['label', 'text']))
     test.append(pd.DataFrame(columns = ['label', 'text']))
     for j in train_index:
         train[i] = train[i].append({'label' : data.label[j], 'text' : data.text[j]}, ignore_index=True)
     for k in test_index:
         test[i] = test[i].append({'label' : data.label[k], 'text' : data.text[k]}, ignore_index=True)
     train[i].to_csv("train" + str(i) + ".csv", header = None, index = False)
     test[i].to_csv("test" + str(i) + ".csv", header = None, index = False)
