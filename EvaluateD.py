import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, LeavePOut, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, auc
from AHFS import AHFS_Algorithm
from Tools import ReadDataSet, GetBestFeatureByMrmr, GetBestFeatureByf_classif, GetBestFeatureByf_RFE, \
    GetBestFeatureByf_ReliefF, readDividedCSV, readCSVClassesInSecondRow, readCSVClassesInSecondRow2, \
    readXlsxClassesInSecondRow, readCSVClassesInLastColumn, OpenFileAndwriteExcelHeader, writeExcelRow
from FFS import FSS_Algorithm
from sklearn import preprocessing

# All the data list with the parameters with the best parameter for the feature selection
# : Features,Reading data set method, records, CV method, filtering algorithm,measurement
datasetList = {
    'DataSet/mat/lung_small.mat': [325, ReadDataSet, 73, 'LOOCV', 'MRMR', 20, 'LogisticsRegression', 'ACC'],
    'DataSet/mat/colon1.mat': [2000, ReadDataSet, 62, 'LOOCV', 'FSS', 4, 'SVM', 'ACC'],
    'DataSet/mat/Isolet.mat': [617, ReadDataSet, 1560, '5FOLDS', 'MRMR', 3, 'NB', 'AUC'],
    'DataSet/mat/madelon.mat': [500, ReadDataSet, 2600, '5FOLDS', 'FSS', 30, 'K_neighbors', 'ACC'],
    'DataSet/mat/ORL1.mat': [1024, ReadDataSet, 400, '10FOLDS', 'FSS', 4, 'LogisticsRegression', 'AUC'],
    'DataSet/mat/pone.0202167.s013.mat': [2000, ReadDataSet, 62, 'LOOCV', 'FSS', 4, 'SVM', 'ACC'],
    'DataSet/mat/USPS.mat': [256, ReadDataSet, 9298, '5FOLDS', 'FSS', 1, 'LogisticsRegression', 'AUC'],
    'DataSet/mat/warpAR10P1.mat': [2400, ReadDataSet, 130, '10FOLDS', 'relief F', 5, 'LogisticsRegression', 'AUC'],
    'DataSet/mat/Yale1.mat': [1024, ReadDataSet, 165, '10FOLDS', 'FSS', 4, 'LogisticsRegression', 'AUC'],
    'DataSet/mat/COIL20.mat': [1024, ReadDataSet, 1440, '5FOLDS', 'FSS', 1, 'LogisticsRegression', 'AUC'],
    'DataSet/divided_csf/sorlie_inputs.csv': [456, readDividedCSV, 85, 'LOOCV', 'MRMR', 100, 'SVM', 'ACC'],
    'DataSet/divided_csf/christensen_inputs.csv': [1413, readDividedCSV, 217, '10FOLDS', 'AHFS', 10, 'SVM', 'AUC'],
    'DataSet/divided_csf/alon_inputs.csv': [2000, readDividedCSV, 62, 'LOOCV', 'FSS', 15, 'RandomForest', 'ACC'],
    'DataSet/Csv/curatedOvarianData.csv': [3584, readCSVClassesInSecondRow, 194, '10FOLDS', 'FSS', 4, 'RandomForest',
                                           'AUC'],
    'DataSet/Csv/DLBCL.csv': [3583, readCSVClassesInSecondRow, 194, '10FOLDS', 'AHFS', 20, 'RandomForest', 'AUC'],
    'DataSet/Csv/NCI60_Ross.csv': [1375, readCSVClassesInSecondRow2, 60, 'LOOCV', 'AHFS', 100, 'SVM', 'ACC'],
    'DataSet/Csv/pone.0246039.s001.csv': [3569, readCSVClassesInLastColumn, 72, 'LOOCV', 'FSS', 2, 'SVM', 'AUC'],
    'DataSet/xlsx/Nutt-2003-v2_BrainCancer.xlsx': [1070, readXlsxClassesInSecondRow, 28, 'LPO', 'MRMR', 10, 'SVM',
                                                   'ACC'],
    'DataSet/xlsx/Risinger_Endometrial Cancer.xlsx': [1771, readXlsxClassesInSecondRow, 42, 'LPO', 'RFE', 3, 'SVM',
                                                      'AUC'],
    'DataSet/xlsx/Singh-2002_ProstateCancer.xlsx': [339, readXlsxClassesInSecondRow, 102, '10FOLDS', 'MRMR', 15, 'NB',
                                                    'ACC'],
}
# the best feature indexes for each  data set
bestFeaturesByDataBase = {
    'DataSet/mat/lung_small.mat': [29, 137, 44, 237, 125, 19, 242, 10, 46, 75, 243, 22, 18, 166, 67, 35, 80, 248, 23,
                                   96],
    'DataSet/mat/colon1.mat': [0, 1, 20, 24],
    'DataSet/mat/Isolet.mat': [91, 417, 436],
    'DataSet/mat/madelon.mat': [2, 56, 0, 4, 3, 109, 1, 121, 7, 16, 8, 17, 5, 188, 6, 9, 12, 29, 14, 42, 10, 18, 11, 15,
                                13, 19, 20, 21, 22, 23],
    'DataSet/mat/ORL1.mat': [0, 5, 61, 32],
    'DataSet/mat/pone.0202167.s013.mat': [0, 1, 20, 24],
    'DataSet/mat/USPS.mat': [135],
    'DataSet/mat/warpAR10P1.mat': [20, 60, 68, 73, 193],
    'DataSet/mat/Yale1.mat': [3, 26, 31, 90],
    'DataSet/mat/COIL20.mat': [451],
    'DataSet/divided_csf/sorlie_inputs.csv': [0, 94, 185, 1, 23, 2, 7, 3, 39, 4, 20, 6, 22, 5, 9, 16, 8, 56, 13, 52, 12,
                                              11, 14, 24, 10, 32, 85, 15, 58, 17, 25, 26, 19, 55, 50, 38, 31, 18, 46,
                                              40, 112, 29, 34, 30, 51, 54, 47, 21, 78, 35, 45, 28, 75, 150, 27, 77, 33,
                                              48, 42, 89, 37, 136, 41, 59, 65, 86, 43, 36, 173, 44, 79, 66, 53, 71, 68,
                                              127, 163, 80, 64, 162, 93, 70, 62, 49, 67, 69, 92, 63, 146, 100, 73, 76,
                                              148, 87, 88, 60, 106, 103, 74, 108],
    'DataSet/divided_csf/christensen_inputs.csv': [16, 193, 5, 10, 184, 22, 46, 225, 0, 32],
    'DataSet/divided_csf/alon_inputs.csv': [0, 57, 2, 43, 3, 124, 6, 13, 7, 35, 8, 69, 17, 27, 18],
    'DataSet/Csv/curatedOvarianData.csv': [0, 1, 2, 21],
    'DataSet/Csv/DLBCL.csv': [57, 19, 43, 360, 2, 381, 69, 97, 39, 354, 37, 83, 240, 357, 5, 190, 110, 176, 9, 45],
    'DataSet/Csv/NCI60_Ross.csv': [31, 68, 112, 48, 115, 60, 328, 72, 7, 45, 11, 191, 319, 1, 125, 2, 5, 473, 135, 76,
                                   92, 77, 78, 39, 32, 70, 225, 59, 3, 86, 4, 139, 42, 89, 9, 50, 44, 138, 35, 166, 19,
                                   25, 418, 51, 152, 179, 27, 17, 40, 116, 136, 18, 242, 209, 358, 301, 197, 251, 121,
                                   80, 142, 0, 117, 46, 8, 199, 105, 88, 311, 71, 73, 230, 346, 171, 317, 459, 23, 21,
                                   347, 123, 263, 449, 211, 204, 193, 13, 488, 275, 438, 445, 127, 494, 37, 206, 140,
                                   114, 208, 95, 106, 58],
    'DataSet/Csv/pone.0246039.s001.csv': [3, 5],
    'DataSet/xlsx/Nutt-2003-v2_BrainCancer.xlsx': [0, 269, 271, 12, 47, 18, 5, 6, 2, 15],
    'DataSet/xlsx/Risinger_Endometrial Cancer.xlsx': [1, 2, 27],
    'DataSet/xlsx/Singh-2002_ProstateCancer.xlsx': [0, 211, 2, 1, 10, 4, 3, 17, 5, 6, 50, 15, 22, 9, 125],
}

'''
# Change the records to be with only the best given feature indexes.
:param Records - Data set records
:param BestFeatures - Best features indexes
'''


def GetRecordsByBestFeatures(Records, BestFeatures):
    newRecordsMat = list()
    for record in Records:
        modifiedRecord = list()
        for featureIndex in BestFeatures:
            modifiedRecord.append(record[featureIndex])
        newRecordsMat.append(modifiedRecord)
    return np.array(newRecordsMat)


'''
# Get Best K Features based on the given model
:param model - Example: K_neighbors,LogisticsRegression
'''


def modelOfFeatureSelectionD(X_train, X_test, y_train, y_test, model, isMulti):
    clf = 0
    match model:
        case 'NB':
            clf = GaussianNB()
        case 'SVM':
            clf = svm.SVC(probability=True)
        case 'LogisticsRegression':
            clf = LogisticRegression()
        case 'RandomForest':
            clf = RandomForestClassifier()
        case 'K_neighbors':
            clf = KNeighborsClassifier()
    clf = clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    # ** compute ACC ****
    acc = balanced_accuracy_score(y_test, prediction)
    # print('got acc ', acc)

    # ** compute AUC ****
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction, pos_label=2)
    aucOut = metrics.auc(fpr, tpr)
    # print('got auc ', aucOut)
    # ** compute MCC ****
    mcc = matthews_corrcoef(y_test, prediction)
    # print('got mcc ', mcc)

    return acc, mcc, aucOut


'''
1.Split the data set to train and test.
2. Use BorderlineSMOTE to smooth the data set.
3. Evaluate their accuracy based the the cv method.
:param CVmethod - Example: LOOCV,10FOLDS
:paramm odel - Example: RandomForest
:param BestKFeaturs - Best feature indexes.
'''


def SmothAndEvaultionModel(CVmethod, model, Records, Labels):
    ACC = 0
    MCC = 0
    AUC = 0
    CV = 0
    match CVmethod:
        case 'LPO':
            CV = LeavePOut(2)
            numOfFolds = 2
        case 'LOOCV':
            CV = LeaveOneOut()
            numOfFolds = 1
        case '5FOLDS':
            CV = KFold(n_splits=5, shuffle=True)
            numOfFolds = 5
        case '10FOLDS':
            CV = KFold(n_splits=10, shuffle=True)
            numOfFolds = 10

    CV.get_n_splits(Records)
    count = 0
    counterForAUC = 0
    for train_index, test_index in CV.split(Records):
        X_train, X_test = Records[train_index], Records[test_index]
        y_train, y_test = Labels[train_index], Labels[test_index]

        oversampler = BorderlineSMOTE(k_neighbors=3)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

        isUniqu = set(Labels)
        acc, mcc, aucOut = modelOfFeatureSelectionD(X_train, X_test, y_train, y_test, model, len(isUniqu) > 2)
        ACC += acc
        MCC += mcc
        if not np.isnan(aucOut):
            AUC += aucOut
            counterForAUC += 1
        count += 1
    if counterForAUC == 0:
        counterForAUC = 1
    return ACC / count, MCC / count, AUC / counterForAUC


# Contact the array.
def addThreeArr(ArrOrigin, Arr1, Arr2):
    out = list()
    for item in ArrOrigin:
        out.append(item)
    for item in Arr1:
        out.append(item)
    for item in Arr2:
        out.append(item)
    return np.array(out)


# contact 3 2d array
def combineAllRecords(Records, X_transformed_linear, X_transformed_Rbf):
    newRecords = list()
    for i in range(0, len(Records)):
        tempRec = addThreeArr(Records[i], X_transformed_linear[i], X_transformed_Rbf[i])
        newRecords.append(tempRec)
    return np.array(newRecords)


'''
Go over the files and for each file evaluate the data set with the following:
1. Algorithm - Feature selection algorithm. Example: mrmr,FSS.
2. K - Best K features. Example: 1,2,5...
3. Model - Learning algorithm. Example: RandomForest,K_neighbors
4. Measurement - Measurement method. Example: Auc, ACC
'''


def EvaluteAllFilesD():
    for file in datasetList.keys():
        try:

            indexForExel = 1
            Records, Labels = datasetList[file][1](file)
            # if len(Records) > 700:
            #     Records = Records[0:700]
            #     Labels = Labels[0:700]
            le = preprocessing.LabelEncoder()
            le.fit(Labels)
            Labels = le.transform(Labels)
            if Records[0] is not int:
                Records = [list(map(float, i)) for i in Records]
            CVmethod = datasetList[file][3]
            model = datasetList[file][6]
            measureType = datasetList[file][7]
            BestFeatures = bestFeaturesByDataBase[file]
            Records = GetRecordsByBestFeatures(Records, BestFeatures)
            transformerLinear = KernelPCA(kernel='linear')
            transformerRbf = KernelPCA(kernel='rbf')
            X_transformed_linear = transformerLinear.fit_transform(Records)
            X_transformed_Rbf = transformerRbf.fit_transform(Records)
            Records = combineAllRecords(Records, X_transformed_linear, X_transformed_Rbf)

            ACC, MCC, AUC = SmothAndEvaultionModel(CVmethod, model, Records, Labels)
            measure = 0
            match measureType:
                case 'ACC':
                    measure = ACC
                case 'AUC':
                    measure = AUC
                case 'MCC':
                    measure = MCC
            print(file, measureType, measure)
        except:
            print(file, 'exeption')


if __name__ == '__main__':
    EvaluteAllFilesD()
