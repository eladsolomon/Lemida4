import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeavePOut, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from AHFS import AHFS_Algorithm
from Tools import ReadDataSet, GetBestFeatureByMrmr, GetBestFeatureByf_classif, GetBestFeatureByf_RFE, \
    GetBestFeatureByf_ReliefF, OpenFileAndwriteExcelHeader, writeExcelRow
from FFS import FSS_Algorithm, New_FSS_Algorithm
from sklearn import preprocessing

# All the data list with the parameters: Features,Reading data set method, records, CV method
datasetList = {
    'DataSet/mat/lung_small.mat': [325, ReadDataSet, 73, 'LOOCV'],
    # 'DataSet/mat/colon1.mat': [2000, ReadDataSet, 62, 'LOOCV'],
    # 'DataSet/mat/Isolet.mat': [617, ReadDataSet, 1560, '5FOLDS'],
    # 'DataSet/mat/madelon.mat': [500, ReadDataSet, 2600, '5FOLDS'],
    # 'DataSet/mat/ORL1.mat': [1024, ReadDataSet, 400, '10FOLDS'],
    # 'DataSet/mat/pone.0202167.s013.mat': [2000, ReadDataSet, 62, 'LOOCV'],
    #  'DataSet/mat/USPS.mat': [256, ReadDataSet, 9298, '5FOLDS'],
    # 'DataSet/mat/warpAR10P1.mat': [2400, ReadDataSet, 130, '10FOLDS'],
    # 'DataSet/mat/Yale1.mat': [1024, ReadDataSet, 165, '10FOLDS'],Done
    # 'DataSet/mat/COIL20.mat': [1024, ReadDataSet, 1440, '5FOLDS'],Done
    # 'DataSet/divided_csf/sorlie_inputs.csv': [456, readDividedCSV, 85, 'LOOCV'],
    # 'DataSet/divided_csf/christensen_inputs.csv': [1413, readDividedCSV, 217, '10FOLDS'],
    # 'DataSet/divided_csf/alon_inputs.csv': [2000, readDividedCSV, 62, 'LOOCV'],Done
    # 'DataSet/Csv/curatedOvarianData.csv': [3584, readCSVClassesInSecondRow, 194, '10FOLDS'],
    # 'DataSet/Csv/DLBCL.csv': [3583, readCSVClassesInSecondRow, 194, '10FOLDS'],
    # 'DataSet/Csv/NCI60_Ross.csv': [1375, readCSVClassesInSecondRow2, 60, 'LOOCV'],
    # 'DataSet/Csv/pone.0246039.s001.csv': [3569, readCSVClassesInLastColumn, 72, 'LOOCV'],
    # 'DataSet/xlsx/Nutt-2003-v2_BrainCancer.xlsx': [1070, readXlsxClassesInSecondRow, 28, 'LPO'],
    # 'DataSet/xlsx/Risinger_Endometrial Cancer.xlsx': [1771, readXlsxClassesInSecondRow, 42, 'LPO'],
    # 'DataSet/xlsx/Singh-2002_ProstateCancer.xlsx': [339, readXlsxClassesInSecondRow, 102, '10FOLDS'],
}

'''
# Get Best K Features based on the given model
:param model - Example: K_neighbors,LogisticsRegression
'''


def modelOfFeatureSelection(chosenFeatures, X_train, X_test, y_train, y_test, model, isMulti):
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
            clf = svm.SVC(probability=True)
        case 'LogisticsRegression':
            clf = LogisticRegression()
        case 'RandomForest':
            clf = RandomForestClassifier()
        case 'K_neighbors':
            clf = KNeighborsClassifier()
    clf = clf.fit(newRecordsMat_X_train, y_train)
    prediction = clf.predict(newRecordsMat_X_test)

    # ** compute ACC ****
    acc = balanced_accuracy_score(y_test, prediction)

    # ** compute AUC ****
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction, pos_label=2)
    aucOut = metrics.auc(fpr, tpr)
    # ** compute MCC ****
    mcc = matthews_corrcoef(y_test, prediction)

    return acc, mcc, aucOut


'''
# Get Best K Features based on the given algorithm
:param algorithm - Example: AHFS,mrmr
:param K - Get best K features
'''


def getBestKFeaturesByAlgorithm(algorithm, K, Records, Labels):
    BestFeatures = []
    match algorithm:
        case 'AHFS':
            topKScores, BestFeatures = AHFS_Algorithm(Records, Labels, K)
        case 'FSS':
            topKScores, BestFeatures = FSS_Algorithm(Records, Labels, K)
        case 'mrmr':
            topKScores, BestFeatures = GetBestFeatureByMrmr(Records, Labels, K)
        case 'f_classif':
            topKScores, BestFeatures = GetBestFeatureByf_classif(Records, Labels, K)
        case 'RFE':
            topKScores, BestFeatures = GetBestFeatureByf_RFE(Records, Labels, K)
        case 'ReliefF':
            topKScores, BestFeatures = GetBestFeatureByf_ReliefF(Records, Labels, K)
        case 'New_FSS_Algorithm':
            topKScores, BestFeatures = New_FSS_Algorithm(Records, Labels, K)
    return topKScores, BestFeatures


'''
Split the data set to train and test and evaluate their accuracy based the the cv method.
1. CVmethod - Example: LOOCV,10FOLDS
2. model - Example: RandomForest
3. BestKFeaturs - Best feature indexes.
'''


def modelEvaultion(CVmethod, model, BestKFeaturs, Records, Labels):
    ACC = 0
    MCC = 0
    AUC = 0
    CV = 0
    numOfFolds = 0
    # Choose the cv method
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
    # Spilt the data set to train and test
    for train_index, test_index in CV.split(Records):
        X_train, X_test = Records[train_index], Records[test_index]
        y_train, y_test = Labels[train_index], Labels[test_index]
        isUniqu = set(Labels)
        acc, mcc, aucOut = modelOfFeatureSelection(BestKFeaturs, X_train, X_test, y_train, y_test, model,
                                                   len(isUniqu) > 2)
        ACC += acc
        MCC += mcc
        if not np.isnan(aucOut):
            AUC += aucOut
            counterForAUC += 1
        count += 1
    if counterForAUC == 0:
        counterForAUC = 1
    return ACC / count, MCC / count, AUC / counterForAUC, numOfFolds


'''
Get the beast 1000 feature based f_classif (for reducing running time)
:return: best features by f_classif
'''


def get1000BestFeatures(Records, Labels):
    topKScores, topKFeatures = GetBestFeatureByf_classif(Records, Labels, K=1000)
    newRecordsMat = list()
    for record in Records:
        modifiedRecord = list()
        for featureIndex in topKFeatures:
            modifiedRecord.append(record[featureIndex])
        newRecordsMat.append(modifiedRecord)
    return np.array(newRecordsMat)


# algorithms = ['AHFS', 'FSS', 'mrmr', 'f_classif', 'RFE', 'ReliefF', New_FSS_Algorithm]
algorithms = ['FSS','New_FSS_Algorithm']
models = ['NB', 'SVM', 'LogisticsRegression', 'RandomForest', 'K_neighbors']
Ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
K = 100
'''
Go over the files and for each file evaluate the data set with the following:
1. Algorithm - Feature selection algorithm. Example: mrmr,FSS.
2. K - Best K features. Example: 1,2,5...
3. Model - Learning algorithm. Example: RandomForest,K_neighbors
4. Measurement - Measurement method. Example: Auc, ACC
'''


def EvaluteAllFiles():
    for file in datasetList.keys():
        indexForExel = 1
        fileName = file.split('/')[2].split('.')[0] + '.csv'
        wb, sheet1 = OpenFileAndwriteExcelHeader(fileName)

        Records, Labels = datasetList[file][1](file)

        # Pre processing: 1.Change string labels/records to float/int. 2. If the features number is higher than 1000
        # extract the best 1000 features first(Because the running time can be really long).
        le = preprocessing.LabelEncoder()
        le.fit(Labels)
        Labels = le.transform(Labels)
        if Records[0] is not int:
            Records = [list(map(float, i)) for i in Records]
        Records = np.array(Records)


        if len(Records[0]) > 1000:
            Records = get1000BestFeatures(Records, Labels)

        print('start', file, len(Records[0]), len(Records))
        CVmethod = datasetList[file][3]
        # Go over the algorithms
        for algo in algorithms:
            # Take the best K features by their indexes
            # BestFeatures is array of K feature indexes
            topKScores, BestFeatures = getBestKFeaturesByAlgorithm(algo, K, Records, Labels)
            # Go over all the K
            for currentK in Ks:
                BestKFeaturs = BestFeatures[0:currentK]
                BestKScores = topKScores[0:currentK]
                # Go over all the models
                for model in models:
                    ACC, MCC, AUC, numOfFolds = modelEvaultion(CVmethod, model, BestKFeaturs, Records, Labels)
                    # Write the results to the excel
                    writeExcelRow(wb, sheet1, fileName, indexForExel, file, datasetList[file][2],
                                  datasetList[file][0],
                                  algo, model, currentK, datasetList[file][3], numOfFolds, 'ACC', ACC,
                                  BestKFeaturs, BestKScores)
                    indexForExel += 1
                    writeExcelRow(wb, sheet1, fileName, indexForExel, file, datasetList[file][2],
                                  datasetList[file][0],
                                  algo, model, currentK, datasetList[file][3], numOfFolds, 'AUC', AUC,
                                  BestKFeaturs, BestKScores)
                    indexForExel += 1
                    writeExcelRow(wb, sheet1, fileName, indexForExel, file, datasetList[file][2],
                                  datasetList[file][0],
                                  algo, model, currentK, datasetList[file][3], numOfFolds, 'MCC', MCC,
                                  BestKFeaturs, BestKScores)
                    indexForExel += 1
            print("Finish algo=", algo)
        print('Finish data set:', file, len(Records[0]))


if __name__ == '__main__':
    EvaluteAllFiles()
