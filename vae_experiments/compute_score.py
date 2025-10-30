import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

def prediction_adjust(prediction, labels):
    labels = labels[:len(prediction)]
    i = 0
    length = len(labels)
    while i < length:
        if labels[i] == True:
            j = i

            adjust_flag = False
            while labels[j] == True and j < length:
                if prediction[j] == True:
                    adjust_flag = True
                j += 1
                if j == length:
                    break
            if adjust_flag:
                for k in range(i, j):
                    prediction[k] = True
            i = j
        else:
            i += 1
    return prediction

def calculate_metrics( y_true, y_pred):
    accuracy = np.round(accuracy_score(y_true, y_pred), 4)
    precision = np.round(precision_score(y_true, y_pred), 4)
    recall = np.round(recall_score(y_true, y_pred), 4)
    f1 = np.round(f1_score(y_true, y_pred), 4)
    roc_auc = np.round(roc_auc_score(y_true, y_pred), 4)

    return accuracy, precision, recall, f1, roc_auc