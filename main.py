from numpy.core.fromnumeric import prod
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_ilpd():
    data = pd.read_csv('indian_liver_patient.csv')
    data['Gender'] = data['Gender'].apply(lambda s: 0 if s == "Male" else 1)
    del data["Age"]
    # convfunc = lambda s: 0 if np.char.equal(s.decode('UTF-8'), "Male") else 1
    # data = np.loadtxt('indian_liver_patient.csv', skiprows=1 ,delimiter=',', converters={1: convfunc})
    return {"data": data.drop("Dataset", axis=1).to_numpy(), "target": data["Dataset"].to_numpy(), "feature_names": data.drop("Dataset", axis=1).columns.tolist()}

class NaiveBiayesClassifier:

    #Target column, with target classes, has to be named "target"
    def fit(self, data: pd.DataFrame):
        self.data = data
        self.targets = data["target"].to_numpy()
        self.names = data.drop("target", axis=1).columns.values

        #prior probabilities
        self.prior_proba = data.groupby('target').size().div(len(data))

        #likelihood of different features for each class
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
            #Find closesed sample to the given and take it index in dataframe
            closest = np.abs(self.data[self.names[i]].to_numpy() - sample[i]).argmin()

            # #TODO Here maby error
            proba_P = np.copy(self.proba[self.names[i]][0])
            proba_h = np.copy(self.proba[self.names[i]][1])

            for j in range(len(proba_h)):
                proba_h[j] = proba_h[j][0]

            #Find closesed sample to the given and take it index in propabiliti distribution dataset
            idx = np.abs(proba_h - np.linalg.norm(proba_h - self.data[self.names[i]].to_numpy()[closest])).argmin()
            #idx = np.abs(proba_h - self.data[self.names[i]].to_numpy()[closest]).argmax()


            #{0.0: [0.001,0.001,0.001,0.001], 2.0: [0.001,0.001,0.001,0.001], 1.0: [0.001,0.001,0.001,0.001]}
            answ[self.targets[closest]] = answ[self.targets[closest]] + proba_P[idx]
            #{0.0: [0.13953488,0.001,0.27906977,0.53488372], 2.0: [0.001,0.001,0.001,0.001], 1.0: [0.001,0.18604651,0.001, 0.001]}

        #p(D)*Пp(h|D)
        for pDN_idx, pN in zip(answ, self.prior_proba.to_numpy()):
            answ[pDN_idx] = pN*np.prod(answ[pDN_idx])
        
        return answ


#PLOTTING

#plt.xlim(50, 90)

# plt.ylim(0, 1)
# plt.ylabel("Точность %")
# plt.xlabel("% обуч. выборки")

#--------

#np.random.seed(46)

data = load_breast_cancer()

col = np.append(data['feature_names'], 'target')

df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                     columns= col)

df = df.sample(frac=1)

std_scl = StandardScaler()
std_scl.fit(df)
scl_df = pd.DataFrame(std_scl.transform(df), index=df.index, columns=df.columns)

nbc = NaiveBiayesClassifier()

acc_arr = []
percent_arr = []
for precent in range(5, 10):
    msk = np.random.rand(len(scl_df)) < precent/10
    train = scl_df[msk]
    test = scl_df[~msk]

    nbc.fit(train)

    count = 0
    for index, row in test.iterrows():
        answ = nbc.predict(row.to_numpy()[:-1])
        if max(answ, key=answ.get) == row.to_numpy()[-1:][0]:
            count = count + 1
    acc_arr.append(count/len(test))
    percent_arr.append(precent/10)
    print("Accuracy: {0}\t Percent: {1}".format(count/len(test), precent))

# plt.xticks(percent_arr)
# plt.plot(percent_arr, acc_arr, label="breast_cancer")
# plt.legend()
# plt.savefig("breast_cancer.png")