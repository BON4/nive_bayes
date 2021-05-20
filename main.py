from numpy.core.fromnumeric import prod
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

class NaiveBiayesClassifier:

    #Target column, with target classes, has to be named "target"
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.prior_proba = data.groupby('target').size().div(len(data))
        self.targets = data["target"].to_numpy()
        self.names = data.drop("target", axis=1).columns.values

        self.proba = {i:[] for i in self.names}
        for i in self.proba:
            t = self._get_proba(i)
            self.proba[i] = [t.to_numpy(), t.index.to_numpy()]

    def _get_proba(self, name: str) ->pd.Series:
        return self.data.groupby([name, 'target']).size().div(len(self.data)).div(self.prior_proba, axis=0, level='target')
    
    def predict(self, sample: np.ndarray):
        answ = {j:np.ones(len(self.names))/1000 for j in self.targets}
        for i in range(len(sample)):
            #Becouse of dis bad accuracy
            closest = np.abs(self.data[self.names[i]].to_numpy() - sample[i]).argmin()

            # #TODO Here maby error
            proba_P = np.copy(self.proba[self.names[i]][0])
            proba_h = np.copy(self.proba[self.names[i]][1])

            for j in range(len(proba_h)):
                proba_h[j] = proba_h[j][0]

            idx = np.abs(proba_h - self.data[self.names[i]].to_numpy()[closest]).argmin()

            answ[self.targets[closest]] = answ[self.targets[closest]] + proba_P[idx]
        
        for pDN_idx, pN in zip(answ, self.prior_proba.to_numpy()):
            answ[pDN_idx] = pN*np.prod(answ[pDN_idx])
        
        return answ

data = load_breast_cancer()

col = np.append(data['feature_names'], 'target')

df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                     columns= col)

df = df.sample(frac=1)

msk = np.random.rand(len(df)) < 0.9
train = df[msk]
test = df[~msk]

nbc = NaiveBiayesClassifier(train)


count = 0
for index, row in test.iterrows():
    answ = nbc.predict(row.to_numpy()[:-1])
    if max(answ, key=answ.get) == row.to_numpy()[-1:][0]:
        count = count + 1

print("Accuracy: ", count/len(test))