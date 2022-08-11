import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import copy

'''
Get the beast feature based the given model
:param ModelName: the model name that will extract the features
:param records data set records:
:param labels data set labels:
:return: best feature 
'''


def GetBestFeatureByModel(ModelName, records, labels):
    clf = 0
    match ModelName:
        case 'ExtraTreesClassifier':
            clf = ExtraTreesClassifier()
        case 'RandomForestClassifier':
            clf = RandomForestClassifier()
        case 'GradientBoostingClassifier':
            clf = GradientBoostingClassifier()
        case 'RandomForestRegressor':
            clf = RandomForestRegressor()
    # sort and get the best feature
    clf = clf.fit(records, labels)
    selected_features = clf.feature_importances_
    var = selected_features.argsort()[-1:][::-1]
    return var[0]


'''
Get the best feature until now.
:param currentSelectedFeaturesIndexes: Current selected features indexes until now.
:param records: data set records
:param labels: data set labels
:param NotRealCurrentSelectedFeatures: Current selected features indexes in the new metric
 (after extraction the selected features) until now.
:return: best features 
'''


def GetCurrentBestFeature(currentSelectedFeaturesIndexes, records, labels, NotRealCurrentSelectedFeatures):
    # Create new records without the records that was chosen until now
    TempRecords = copy.deepcopy(records)
    newRecords = []
    # If its first iteration
    if len(currentSelectedFeaturesIndexes) == 0:
        newRecords = copy.deepcopy(records)
    # Create new metric without the selected features until now
    else:
        for record in records:
            tempRec = list()
            for index in range(0, len(record)):
                if index not in currentSelectedFeaturesIndexes:
                    tempRec.append(record[index])
            newRecords.append(np.array(tempRec))
    newRecords = np.array(newRecords)

    # Get best feature based each algorithms
    ExtraTreesClassifierBestFeatureIndex = GetBestFeatureByModel("ExtraTreesClassifier", newRecords, labels)
    RandomForestClassifierBestFeatureIndex = GetBestFeatureByModel("RandomForestClassifier", newRecords, labels)
    RandomForestRegressorBestFeatureIndex = GetBestFeatureByModel("RandomForestRegressor", newRecords, labels)
    GradientBoostingClassifierBestFeatureIndex = GetBestFeatureByModel("GradientBoostingClassifier", newRecords, labels)
    FeatureSet = set()
    FeatureSet.add(ExtraTreesClassifierBestFeatureIndex)
    FeatureSet.add(RandomForestClassifierBestFeatureIndex)
    FeatureSet.add(GradientBoostingClassifierBestFeatureIndex)
    FeatureSet.add(RandomForestRegressorBestFeatureIndex)
    CurrentBestFeatureIndex = -1
    CurrentBestScore = -1
    # Evaluate and choose the best feature to add
    # Choose the best feature based on all the algorithms
    for uniqueChoosenFeature in FeatureSet:
        CandidateFetures = copy.deepcopy(currentSelectedFeaturesIndexes)
        CandidateFetures.append(uniqueChoosenFeature)
        NewCandidateRecords = []
        for record in TempRecords:
            tempRec = []
            for featureIndex in CandidateFetures:
                tempRec.append(record[featureIndex])
            NewCandidateRecords.append(tempRec)
        NewCandidateRecords = np.array(NewCandidateRecords)
        clf = svm.SVC(kernel='linear')
        clf.fit(np.array(NewCandidateRecords), labels)
        score = clf.score(NewCandidateRecords, labels)
        if score > CurrentBestScore:
            CurrentBestFeatureIndex = uniqueChoosenFeature
            CurrentBestScore = score
    reverseCurrentSelectedFeaturesIndexes = list(reversed(NotRealCurrentSelectedFeatures))
    NotRealIndex = CurrentBestFeatureIndex
    for ind in reverseCurrentSelectedFeaturesIndexes:
        if ind <= CurrentBestFeatureIndex:
            CurrentBestFeatureIndex += 1
    return CurrentBestFeatureIndex, NotRealIndex


'''
Get the best K features by AHFS algorithm
:param K: K best features to get
:param records data set records:
:param labels data set labels:
:return: best features 
'''


def AHFS_Algorithm(records, labels, K=1):
    currentSelectedFeatures = list()
    NotRealCurrentSelectedFeatures = list()
    scores = [1] * K
    while len(currentSelectedFeatures) < K:
        TempFeature = copy.deepcopy(currentSelectedFeatures)
        TempRecords = copy.deepcopy(records)
        BestFeature, NotRealFeature = GetCurrentBestFeature(TempFeature, TempRecords, labels,
                                                            NotRealCurrentSelectedFeatures)
        currentSelectedFeatures.append(BestFeature)
        NotRealCurrentSelectedFeatures.append(NotRealFeature)
    return scores, currentSelectedFeatures
