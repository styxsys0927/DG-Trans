from sklearn import linear_model
from sklearn.svm import SVR
from load_incident import load_incident2
from config import dataset
import numpy as np

dataloader = load_incident2(dataset)
m1, m2 = dataloader['scaler'][0],dataloader['scaler'][1]

for a in [0.0001, 0.001, 0.1, 1.0]:
    clf = linear_model.Lasso(alpha=a)
    clf.fit(dataloader['x_train'], dataloader['y_train'])
    out = clf.predict(dataloader['x_test'])
    y = dataloader['y_test']

    print(m1.inverse_transform(np.sqrt(np.mean((y[:, 0] - out[:, 0]) ** 2)))/60,
          m2.inverse_transform(np.sqrt(np.mean((y[:, 1] - out[:, 1]) ** 2)))/1600,
          m1.inverse_transform(np.mean(np.abs(y[:, 0] - out[:, 0])))/60,
          m2.inverse_transform(np.mean(np.abs(y[:, 1] - out[:, 1])))/1600,
          np.mean(np.abs((y[:, 0]-out[:,0])/(np.abs(y[:, 0])+np.abs(out[:,0]))*2)),
          np.mean(np.abs((y[:, 1]-out[:,1])/(np.abs(y[:, 1])+np.abs(out[:,1]))*2)),
          1 - np.sum((y[:, 0] - out[:, 0]) ** 2) / np.sum((y[:, 0] - np.mean(y[:, 0])) ** 2),
          1 - np.sum((y[:, 1] - out[:, 1]) ** 2) / np.sum((y[:, 1] - np.mean(y[:, 1])) ** 2)
          )

# for c in [0.1]:
#     clf1 = SVR(C=1.0, epsilon=0.1)
#     clf1.fit(dataloader['x_train'], dataloader['y_train'][:,0])
#     clf2 = SVR(C=1.0, epsilon=0.1)
#     clf2.fit(dataloader['x_train'], dataloader['y_train'][:,1])
#     out = np.stack([clf1.predict(dataloader['x_test']), clf2.predict(dataloader['x_test'])], axis=-1)
#     y = dataloader['y_test']
#     print(m1.inverse_transform(np.sqrt(np.mean((y[:, 0] - out[:, 0]) ** 2)))/60,
#           m2.inverse_transform(np.sqrt(np.mean((y[:, 1] - out[:, 1]) ** 2)))/1600,
#           m1.inverse_transform(np.mean(np.abs(y[:, 0] - out[:, 0])))/60,
#           m2.inverse_transform(np.mean(np.abs(y[:, 1] - out[:, 1])))/1600,
#           np.mean(np.abs((y[:, 0]-out[:,0])/(np.abs(y[:, 0])+np.abs(out[:,0]))*2)),
#           np.mean(np.abs((y[:, 1]-out[:,1])/(np.abs(y[:, 1])+np.abs(out[:,1]))*2)),
#           )
