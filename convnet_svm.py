import cPickle
import numpy as np
from sklearn import preprocessing, svm

DO_TRAINING = True
d = '../data-livdet-2015/z_other_features/BSIF-DigPer-2015-features_augmented'

nu = 0.3
gamma = 1e-05
clf_file = d + '/_nu_{:.2g}_{:.2g}.pkl'.format(nu, gamma)

# live=0 fake=1
if DO_TRAINING:
    train_fake = np.loadtxt(d + '/Data_2015_BSIF_7_12_motion_Train_Spoof_DigPerson.txt')
    train_live = np.loadtxt(d + '/Data_2015_BSIF_7_12_motion_Train_Real_DigPerson.txt')
    train_x = np.concatenate((train_fake, train_live))
    train_y = np.concatenate((np.ones(len(train_fake)), np.zeros(len(train_live))))
    del train_fake, train_live
    preprocessing.scale(train_x, copy=False)

    # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm
    clf = svm.NuSVC(kernel='rbf', nu=nu, gamma=gamma, cache_size=1024)
    clf.fit(train_x, train_y)
    # with open(clf_file, 'wb') as f:
    #     cPickle.dump(clf, f)
# else:
    test_fake = np.loadtxt(d + '/Data_2015_BSIF_7_12_motion_Test_Spoof_DigPerson.txt')
    test_live = np.loadtxt(d + '/Data_2015_BSIF_7_12_motion_Test_Real_DigPerson.txt')
    test_x = np.concatenate((test_fake, test_live))
    test_y = np.concatenate((np.ones(len(test_fake)), np.zeros(len(test_live))))
    del test_fake, test_live
    preprocessing.scale(test_x, copy=False)

    # with open(clf_file, 'rb') as f:
    #     clf = cPickle.load(f)
    predicted = clf.predict(test_x)

    # fpr=misclassified_live fnr=misclassified_fake
    fpr = float(np.sum((predicted != test_y) & (test_y == 0))) / np.sum(test_y == 0)
    fnr = float(np.sum((predicted != test_y) & (test_y == 1))) / np.sum(test_y == 1)
    ace = (fpr + fnr) / 2
    n_ok = np.sum(predicted == test_y)
    print 'average classification error = {:.2f}'.format(ace * 100)
    print 'validation accuracy = {:.2f} ({:d}/{:d})'.format(float(n_ok) * 100 / len(predicted), n_ok, len(predicted))
