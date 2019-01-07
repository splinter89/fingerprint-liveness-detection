# live=0 fake=1
fpr = float(np.sum((predicted != test_y) & (test_y == 0))) / np.sum(test_y == 0)    # misclassified live
fnr = float(np.sum((predicted != test_y) & (test_y == 1))) / np.sum(test_y == 1)    # misclassified fake
ace = (fpr + fnr) / 2
print(ace * 100)
