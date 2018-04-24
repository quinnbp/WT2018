from confusion_matrix import ConfusionMatrix
from sklearn.metrics import classification_report
from weighting import voting
import numpy as np

def test_confusion_matrix():
    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]

    cm = ConfusionMatrix(yactual, ypred, "Sample_classifier")

    number_cm = cm.get_number_cm()
    normalized_cm = cm.get_normalized_cm()
    metrics = {"TP": cm.get_true_pos(), "TN": cm.get_true_neg(), "FP": cm.get_false_pos(), "FN": cm.get_false_neg()}

    print("Confusion matrix:")
    print(number_cm)

    print("Normalized confusion matrix:")
    print(normalized_cm)

    print("Precision per label is:", cm.get_precision())
    print("Metrics:", metrics)

    print("MCC is", cm.get_mcc())

    return cm.get_precision()

def test_weighting():
    # Create confusion matrices for random classifiers
    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred1 = np.random.randint(3, size=12)

    cm1 = ConfusionMatrix(yactual, ypred1, "cls_1")

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred2 = np.random.randint(3, size=12)

    cm2 = ConfusionMatrix(yactual, ypred2, "cls_2")

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred3 = np.random.randint(3, size=12)

    cm3 = ConfusionMatrix(yactual, ypred3, "cls_3")

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred4 = np.random.randint(3, size=12)

    cm4 = ConfusionMatrix(yactual, ypred4, "cls_4")

    weight_pairs = [[cm1, ypred1], [cm2, ypred2], [cm3, ypred3], [cm4, ypred4]]

    # Check that CEN score is being calculated
    #print(cm1.get_CEN_score(), cm2.get_CEN_score(), cm3.get_CEN_score(), cm4.get_CEN_score())

    # Get final votes based on pairs
    votes = voting(weight_pairs)

    # Check metrics
    print(classification_report(yactual, votes))

    print(votes)


def main():

    #test_confusion_matrix()

    test_weighting()

if __name__ == "__main__":
    main()

