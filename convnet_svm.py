import cPickle
import numpy as np
from sklearn import preprocessing, svm

DO_TRAINING = True
d = '../data-livdet-2015'

nu = 0.3
gamma = 1e-05
clf_file = d + '/_nu_{:.2g}_{:.2g}.pkl'.format(nu, gamma)

# live=0 fake=1
if DO_TRAINING:
    train_x = np.concatenate((np.load(d + '/train_fake.npy'), np.load(d + '/train_live.npy')))
    train_y = np.concatenate((np.ones(1000), np.zeros(1000)))
    preprocessing.scale(train_x, copy=False)

    # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm
    clf = svm.NuSVC(kernel='rbf', nu=nu, gamma=gamma, cache_size=1024)
    clf.fit(train_x, train_y)
    with open(clf_file, 'wb') as f:
        cPickle.dump(clf, f)
else:
    test_x = np.concatenate((np.load(d + '/test_fake.npy'), np.load(d + '/test_live.npy')))
    test_y = np.concatenate((np.ones(1500), np.zeros(1000)))
    preprocessing.scale(test_x, copy=False)

    with open(clf_file, 'rb') as f:
        clf = cPickle.load(f)
    predicted = clf.predict(test_x)

    # fpr=misclassified_live fnr=misclassified_fake
    fpr = float(np.sum((predicted != test_y) & (test_y == 0))) / np.sum(test_y == 0)
    fnr = float(np.sum((predicted != test_y) & (test_y == 1))) / np.sum(test_y == 1)
    ace = (fpr + fnr) / 2
    n_ok = np.sum(predicted == test_y)
    print 'average classification error = {:.2f}'.format(ace * 100)
    print 'validation accuracy = {:.2f} ({:d}/{:d})'.format(float(n_ok) / len(predicted), n_ok, len(predicted))
