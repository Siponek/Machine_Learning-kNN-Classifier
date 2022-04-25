from sklearn.metrics import classification_report, accuracy_score, euclidean_distances
from os.path import exists
from array import array

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import idx2numpy


def calculateAccuracy(y_test, y_pred):
    y_boolDiff = y_test == y_pred
    y_boolDiff = sum(y_boolDiff)

    return y_boolDiff/y_pred.__len__()


def main():
    kValues = []
    # Loading config, datasets, k value
    dataConfig = json.load(open(file="./config.jsonc", encoding="utf-8"))
    kValues = dataConfig["json_kValues"]
    saveNameHeatmap = dataConfig["json_heatmapName"]
    pathToTrainImages = dataConfig["json_pathToTrainImages"]
    pathToTrainLabels = dataConfig["json_pathToTrainLabel"]
    pathToTestImages = dataConfig["json_pathToTestImages"]
    pathToTestLabels = dataConfig["json_pathToTestLabel"]
    targetPresent = True

    # loading datasets into variables and checking for errors
    try:
        x_train = idx2numpy.convert_from_file(pathToTrainImages)
        x_train = x_train.reshape([x_train.shape[0], 784])

        y_train = idx2numpy.convert_from_file(pathToTrainLabels)

    except Exception as errorCode:
        print(errorCode)
        print("Something is wrong with the training set, make sure it follows the naming convention('t10k-images-idx3-ubyte') and both the training set and truth table are present")
        exit("Error while loading train data.\nExiting the program")

    if not exists(pathToTestImages):
        print(pathToTestImages, "does not exist?")
        exit("Error while loading train data.\nExiting the program")
    else:
        print("Passed image path test")

    try:
        x_test = idx2numpy.convert_from_file(pathToTestImages)
        x_test = x_test.reshape([x_test.shape[0], 784])

    except Exception as errorCode:
        print(errorCode)
        print("Something is wrong with the xtest set, make sure it follows the naming convention('t10k-images-idx3-ubyte') and the test set is present")

    try:
        y_test = idx2numpy.convert_from_file(pathToTestLabels)

    except Exception as errorCode:
        print(errorCode)
        print("Something is wrong with the ytest set, make sure it follows the naming convention('t10k-labels-idx3-ubyte') and both the test set and truth table are present")
        print("Continuing without test set truth table")
        targetPresent = False

    # ?     Check that k>0
    if (len(kValues) < 1 and (all([isinstance(item, int) for item in kValues]))):
        print("Array of kValues specified in config.json must have at least one integer value, bigger than 0")
        exit("Error while loading checking k Value.\nExiting the program")

    # ?     And k<=cardinality of the training set (number of rows, above referred to as n)
    if all([(item == len(x_train)) for item in kValues]):
        print(
            "Array of kValues specified in config.json must have at least one integer value")
        exit("Error while comparing the number of collums of train-test set.\nExiting the program")

    # ?     Check that the number of columns of the second matrix equals the number of columns of the first matrix
    if len(x_train[0]) != len(x_test[0]):
        print(
            "Array of kValues specified in config.json must have at least one integer value")
    # return(print("Finished tests"))
    print("...Finished tests")
    print("Calculating kNN algorythm for these k parameter values:", kValues)
    # return 0

    y_train = pd.DataFrame(data=y_train)
    x_train = pd.DataFrame(data=x_train)
    distanceMatrix = pd.DataFrame(data=(euclidean_distances(x_train, x_test)))
    distanceMatrix.insert(0, column="labelTarget", value=y_train)

    # Limiting sets size for faster test computing
    # disabled by default
    # x_train = x_train[:500]
    # x_test = x_test[:500]
    # y_train = y_train[:500]
    # y_test = y_test[:500]

    dictonaryOfY_pred = {}

    for k in kValues:
        y_pred = np.array([])

        for i in range(x_test.__len__()):

            dfTemp = distanceMatrix.loc[:, ["labelTarget", i]]

            kMinimalPoints = dfTemp.sort_values(by=[i], ascending=True)[:k]
            kMinimalPoints["weight"] = 1/kMinimalPoints[i]
            kMinimalPoints = kMinimalPoints[["labelTarget", "weight"]]
            kMinimalPoints = kMinimalPoints.groupby(["labelTarget"]).sum()
            kMinimalPoints = kMinimalPoints[kMinimalPoints.weight ==
                                            kMinimalPoints.weight.max()]
            kMinimalPoints = kMinimalPoints.reset_index()
            y_pred = np.append(y_pred, kMinimalPoints["labelTarget"])

        if (targetPresent == True):
            accuracy = calculateAccuracy(y_test, y_pred)
            print(
                f"This is the accuracy of prediction {k} on test set {accuracy}")
            dictonaryOfY_pred[k] = {"accuracy": accuracy,
                                    "y_pred":  y_pred}
        else:
            print("These are predictions from the model:")
            y_pred = np.array(y_pred)
            np.savetxt(f"{k}_K_Value.csv", y_pred, delimiter=",")
    if (targetPresent == False):
        return(print("Finised program without a test target set. Results are stored in .csv files"))

    #! Starting a part that plots the accuracy
    plt.figure(dpi=80, figsize=(10, 10))
    # Key of this dict is K value that is specified in task 3
    # Its items are dicts with digits and corresponding accuracy
    finalK_Result = {}

    for key_K, valueDict in dictonaryOfY_pred.items():
        differenceDF = pd.DataFrame(
            list(zip(y_test, y_pred)), columns=["y_test", "y_pred"])
        accuracyResults = {}
        y_pred = valueDict["y_pred"]

        for i in (differenceDF["y_test"].unique()):
            signleDigit = differenceDF[differenceDF["y_test"] == i]
            accuracyResults[i] = calculateAccuracy(
                signleDigit["y_test"], signleDigit["y_pred"])

        finalK_Result[key_K] = accuracyResults
    finalK_Result = pd.DataFrame.from_dict(finalK_Result)
    finalK_Result
    sns_pp = sns.heatmap(data=finalK_Result)
    print("\nSaving results as heatmap to : ", saveNameHeatmap)
    plt.savefig(saveNameHeatmap)

    return(print("Finised program with a test target set"))


print("Starting the program...")

if __name__ == "__main__":
    main()
