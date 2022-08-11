import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import operator


'''
Get all the pair comibinations.
For example:
for numOfFeatures = 4
return: (1,2) (1,3) (1,4) (2,3) (2,4) (3,4)
'''


def getListOfPairsOfIndexes(numOfFeatures):
    indexesPair = list()
    for i in range(0, numOfFeatures):
        for j in range(i + 1, numOfFeatures):
            if i != j:
                indexesPair.append((i, j))
    return indexesPair


'''
Generates a linear line that separates the samples according to their class
and gives a score according to how well the linear line divides the points.
'''


def getTScore(featurePairIndexes, Records, Labels):
    (index1, index2) = featurePairIndexes
    # making new matrix that contains only the chosen features with shape(numOfRecords_X_2)
    newRecordsMat = list()
    for record in Records:
        newRecordsMat.append([record[index1], record[index2]])
    clf = svm.SVC(kernel='linear')
    clf.fit(np.array(newRecordsMat), Labels)
    score = clf.score(newRecordsMat, Labels)
    return score


'''
Generates a poly that separates the samples according to their class
and gives a score according to how well the linear line divides the points.
'''


def New_getTScore(featurePairIndexes, Records, Labels):
    (index1, index2) = featurePairIndexes
    # making new matrix that contains only the chosen features with shape(numOfRecords_X_2)
    newRecordsMat = list()
    for record in Records:
        newRecordsMat.append([record[index1], record[index2]])
    clf = svm.SVC(kernel='poly')
    clf.fit(np.array(newRecordsMat), Labels)
    score = clf.score(newRecordsMat, Labels)
    return score


'''
# Get Best K Features based on the given model
:param model - Example: K_neighbors,LogisticsRegression
'''


def modelOfFeatureSelection(chosenFeatures, X_train, X_test, y_train, y_test, model):
    newRecordsMat_X_train = list()
    for record_x_train in X_train:
        modifiedX_train = list()
        for featureIndex in chosenFeatures:
            modifiedX_train.append(record_x_train[featureIndex])
        newRecordsMat_X_train.append(modifiedX_train)

    newRecordsMat_X_test = list()
    for record_x_test in X_test:
        modifiedX_test = list()
        for featureIndex in chosenFeatures:
            modifiedX_test.append(record_x_test[featureIndex])
        newRecordsMat_X_test.append(modifiedX_test)

    clf = 0
    match model:
        case 'NB':
            clf = GaussianNB()
        case 'SVM':
            clf = svm.SVC()
        case 'LogisticsRegression':
            clf = LogisticRegression()
        case 'RandomForest':
            clf = RandomForestClassifier()
        case 'K_neighbors':
            clf = KNeighborsClassifier()
    clf = clf.fit(newRecordsMat_X_train, y_train)
    prediction = clf.predict(newRecordsMat_X_test)
    acc = balanced_accuracy_score(y_test, prediction)
    return acc


'''
Get the ACC of the data set according to the model evaultion score.
'''


def modelOfOriginDataset(X_train, X_test, y_train, y_test, model):
    clf = 0
    match model:
        case 'NB':
            clf = GaussianNB()
        case 'SVM':
            clf = svm.SVC()
        case 'LogisticsRegression':
            clf = LogisticRegression()
        case 'RandomForest':
            clf = RandomForestClassifier()
        case 'K_neighbors':
            clf = KNeighborsClassifier()
    clf = clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    acc = balanced_accuracy_score(y_test, prediction)
    return acc


'''
Get the best K features by FSS algorithm - using linear line that separates the samples according to their class
and gives a score according to how well the linear line divides the points.
:param K: K best features to get
:param records data set records:
:param labels data set labels:
:return: best features 
'''


def FSS_Algorithm(records, labels, K=10):
    numOfFeatures = len(records[0])

    # get all indexes of pairs combinations
    featuresParis = getListOfPairsOfIndexes(numOfFeatures)
    numFeaturesPairs = len(featuresParis)
    # Evaluate each pair T score
    indexesPair_Tscore = dict()
    count = 1
    for featurePairIndexes in featuresParis:
        # for each pair of indexes get the pair_T_Score
        score = getTScore(featurePairIndexes, records, labels)
        indexesPair_Tscore[featurePairIndexes] = score
        print(round((count / numFeaturesPairs) * 100, 2), count, score)
        count += 1
    # sort the dictionary based on his T value descending order
    sortedDict = dict(sorted(indexesPair_Tscore.items(), key=operator.itemgetter(1), reverse=True))
    # Take the top K features with the highest T
    sortedDictKeys = list(sortedDict.keys())

    # go over the dict and take only the best K features.
    chosenFeatures = list()
    chosenScores = list()
    numOfChosenFeatures = 0
    pairIndex = 0
    while numOfChosenFeatures < K:
        pair = sortedDictKeys[pairIndex]
        pairIndex += 1
        if pair[0] not in chosenFeatures and pair[1] not in chosenFeatures:
            chosenFeatures.append(pair[0])
            chosenFeatures.append(pair[1])
            numOfChosenFeatures += 2
            chosenScores.append(sortedDict[pair])
            chosenScores.append(sortedDict[pair])

    if numOfChosenFeatures > K:
        chosenFeatures = chosenFeatures[:-1]
        chosenScores = chosenScores[:-1]

    return chosenScores, chosenFeatures


'''
Get the best K features by New_FSS_Algorithm algorithm - using poly line that separates the samples according to their class
and gives a score according to how well the linear line divides the points.
:param K: K best features to get
:param records data set records:
:param labels data set labels:
:return: best features 
'''


def New_FSS_Algorithm(records, labels, K):
    labelsLen = len(labels)
    numOfFeatures = len(records[0])
    # print("number of features: ", numOfFeatures)

    # get all indexes of pairs combinations
    featuresParis = getListOfPairsOfIndexes(numOfFeatures)
    numFeaturesPairs = len(featuresParis)
    # Evaluate each pair T score
    indexesPair_Tscore = dict()
    count = 1
    for featurePairIndexes in featuresParis:
        # for each pair of indexes get the pair_T_Score
        score = New_getTScore(featurePairIndexes, records, labels)
        indexesPair_Tscore[featurePairIndexes] = score
        print(round((count / numFeaturesPairs) * 100, 2), count, score)
        count+=1
    # sort the dictionary based on his T value descending order
    sortedDict = dict(sorted(indexesPair_Tscore.items(), key=operator.itemgetter(1), reverse=True))
    # print("Finish evaluate t scores", time.time() - start_time, "to run")

    # Take the top K features with the highest T
    sortedDictKeys = list(sortedDict.keys())

    chosenFeatures = list()
    chosenScores = list()
    numOfChosenFeatures = 0
    pairIndex = 0
    while numOfChosenFeatures < K:
        pair = sortedDictKeys[pairIndex]
        pairIndex += 1
        if pair[0] not in chosenFeatures and pair[1] not in chosenFeatures:
            chosenFeatures.append(pair[0])
            chosenFeatures.append(pair[1])
            numOfChosenFeatures += 2
            chosenScores.append(sortedDict[pair])
            chosenScores.append(sortedDict[pair])

    if numOfChosenFeatures > K:
        chosenFeatures = chosenFeatures[:-1]
        chosenScores = chosenScores[:-1]

    return chosenScores, chosenFeatures
