import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=4)
scaler = StandardScaler()

# Load Data
X_train = np.load('./data/X_train.npy')
X_test = np.load('./data/X_test.npy')
y_train = np.load('./data/y_train.npy')

X_all = np.r_[X_train, X_test]


scaler.fit(X_all)
pca.fit(scaler.fit(X_all))



