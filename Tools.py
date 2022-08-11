import csv
import os
from scipy import stats
import pandas as pd
import scipy.io
import numpy as np
import xlrd
from mrmr import mrmr_classif
from xlwt import Workbook
from sklearn.feature_selection import SelectKBest, f_classif, SelectFdr, RFE
from sklearn.svm import OneClassSVM
from ReliefF import ReliefF
from AHFS import AHFS_Algorithm
from FFS import FSS_Algorithm


############################################################
# Tools that help us with reading and evaluate the data sets.
############################################################

def ReadDataSet(filePath):
    mat = scipy.io.loadmat(filePath)
    # flatter the list
    flat_list = [item for sublist in mat['Y'] for item in sublist]
    return mat['X'], np.array(flat_list)


def GetBestFeatureByMrmr(Records, Labels, K=1):
    X = pd.DataFrame(Records)
    y = pd.Series(Labels)
    BestFeatures_mrmr = mrmr_classif(X=X, y=y, K=K, show_progress=False, return_scores=True)
    FeatureIndexes = BestFeatures_mrmr[0]
    scores = BestFeatures_mrmr[1].values[FeatureIndexes]
    return scores, FeatureIndexes


def GetBestFeatureByf_classif(Records, Labels, K=1):
    selector = SelectFdr(f_classif, alpha=0.01)
    selector.fit_transform(Records, Labels)
    scores = selector.scores_
    temp = np.partition(-scores, K)
    topKScores = -temp[:K]
    arr = np.array(scores)
    topKFeatures = arr.argsort()[-K:][::-1]
    return topKScores, topKFeatures


def GetBestFeatureByf_RFE(Records, Labels, K=1):
    estimator = OneClassSVM(kernel="linear")
    selector = RFE(estimator, n_features_to_select=1, step=1)
    selector = selector.fit(Records, Labels)
    scores = selector.ranking_
    temp = np.partition(-scores, K)
    topKScores = -temp[:K]
    arr = np.array(scores)
    topKFeatures = arr.argsort()[-K:][::-1]
    return topKScores, topKFeatures


def GetBestFeatureByf_ReliefF(Records, Labels, K=10):
    neighobrs = len(Records) - 1
    fs = ReliefF(n_neighbors=neighobrs, n_features_to_keep=K)
    fs.fit_transform(Records, Labels)
    scores = fs.feature_scores
    temp = np.partition(-scores, K)
    topKScores = -temp[:K]
    arr = np.array(scores)
    topKFeatures = arr.argsort()[-K:][::-1]
    return topKScores, topKFeatures


def readDividedCSV(filePath):
    mat = list()

    with open(filePath, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            mat.append(row)
    records = np.array(mat)

    mat = list()
    with open(filePath.replace('inputs', 'outputs'), 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            mat.append(row)
    flat_list = [item for sublist in mat for item in sublist]
    labels = np.array(flat_list)

    return records, labels


def column(matrix, i):
    return [row[i] for row in matrix]


def readCSVClassesInSecondRow(filePath):
    mat = list()

    with open(filePath, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            mat.append(row)
    labels = mat[1]
    labels = np.array(labels[1:])

    newMat = list()
    for i in range(1, len(labels) + 1):
        col = column(mat, i)
        for j in range(0, len(col)):
            if col[j] == 'NA':
                col[j] = 0
        newMat.append(col[2:])
    records = np.array(newMat)

    return records, labels


def readXlsxClassesInSecondRow(filePath):
    df = pd.read_excel(filePath)
    mat = df.values

    labels = mat[0]
    labels = np.array(labels[1:])

    newMat = list()
    for i in range(1, len(labels) + 1):
        col = column(mat, i)
        newMat.append(col[1:])
    records = np.array(newMat)

    return records, labels


def readCSVClassesInSecondRow2(filePath):
    mat = list()

    with open(filePath, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            mat.append(row)
    labels = mat[1]
    labels = np.array(labels[2:])

    newMat = list()
    for i in range(2, len(labels) + 2):
        col = column(mat, i)
        for j in range(0, len(col)):
            if col[j] == 'NA':
                col[j] = 0
        newMat.append(col[2:])
    records = np.array(newMat)

    return records, labels


def readCSVClassesInFirstColumn(filePath):
    with open(filePath, 'r') as file:
        csvreader = csv.reader(file)
        labels = list()
        records = list()
        for row in csvreader:
            labels.append(row[0])
            records.append(row[1:])

    return np.array(records), np.array(labels)


def readCSVClassesInLastColumn(filePath):
    mat = list()

    with open(filePath, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            mat.append(row)
    labels = column(mat, len(mat[0]) - 1)
    labels = np.array(labels[1:])

    newMat = list()
    for i in range(1, len(labels) + 1):
        col = mat[i]
        col = col[1:]  # remove the index
        col = col[:-1]  # remove the class
        newMat.append(col[2:])
    records = np.array(newMat)

    return records, labels


def RunToyExamples():
    records, labels = readCSVClassesInFirstColumn('DataSet/SPECTF.csv')
    Ks = [5, 10, 20]
    for k in Ks:
        Scores_FSS, indexes_FSS = FSS_Algorithm(records, labels, k)
        Scores_AHFS, indexes_AHFS = AHFS_Algorithm(records, labels, k)
        print("FSS with K=", k, "Index:", indexes_FSS, "Scores:", Scores_FSS)
        print("AHFS with K=", k, "Index:", indexes_AHFS, "Scores:", Scores_AHFS)


# Write the Xcel header
def OpenFileAndwriteExcelHeader(FileName):
    # Workbook is created
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'Dataset Name')
    sheet1.write(0, 1, 'Number of samples')
    sheet1.write(0, 2, 'Original Number of features')
    sheet1.write(0, 3, 'Filtering Algorithm')
    sheet1.write(0, 4, 'Learning algorithm')
    sheet1.write(0, 5, 'Number of features selected (K)')
    sheet1.write(0, 6, 'CV Method')
    sheet1.write(0, 7, 'Fold')
    sheet1.write(0, 8, 'Measure Type')
    sheet1.write(0, 9, 'Measure Value')
    sheet1.write(0, 10, 'List of Selected Features Names (Long STRING)')
    sheet1.write(0, 11, 'Selected Features scores')
    wb.save(FileName)

    return wb, sheet1


def changeList(arr):
    out = list()
    for x in arr:
        out.append('Gen' + str(x))
    return out


def writeExcelRow(wb, sheet1, FileName, rowIndex, DatasetName, NumberOfSamples, OriginalNumberOfFeatures,
                  FilteringAlgorithm, LearningAlgorithm, featuresSelected, CVMethod, Fold, MeasureType, MeasureValue,
                  SelectedFeaturesNames, SelectedFeaturesScores):
    sheet1.write(rowIndex, 0, str(DatasetName))
    sheet1.write(rowIndex, 1, str(NumberOfSamples))
    sheet1.write(rowIndex, 2, str(OriginalNumberOfFeatures))
    sheet1.write(rowIndex, 3, str(FilteringAlgorithm))
    sheet1.write(rowIndex, 4, str(LearningAlgorithm))
    sheet1.write(rowIndex, 5, str(featuresSelected))
    sheet1.write(rowIndex, 6, str(CVMethod))
    sheet1.write(rowIndex, 7, str(Fold))
    sheet1.write(rowIndex, 8, str(MeasureType))
    sheet1.write(rowIndex, 9, str(MeasureValue))
    sheet1.write(rowIndex, 10, str(changeList(SelectedFeaturesNames)))
    sheet1.write(rowIndex, 11, str(SelectedFeaturesScores))
    wb.save(FileName)

cwd = os.path.abspath('finished DB')
filesfinished = os.listdir(cwd)
rows = list()
algorithms = ['AHFS', 'FSS', 'mrmr', 'f_classif', 'RFE', 'ReliefF', "New_FSS_Algorithm"]
for file in filesfinished:
    try:
        workbook = xlrd.open_workbook('finished DB\\'+file)
        #Get the first sheet in the workbook by index
        AccAHFS = list()
        AccFSS = list()
        Accmrmr = list()
        Accf_classif = list()
        AccRFE = list()
        AccReliefF = list()
        sheet1 = workbook.sheet_by_index(0)
        #Get each row in the sheet as a list and print the list
        for rowNumber in range(1,sheet1.nrows):
            row = sheet1.row_values(rowNumber)
            match row[3]:
                case 'AHFS':
                    AccAHFS.append(row[9])
                case 'FSS':
                    AccFSS.append(row[9])
                case 'mrmr':
                    Accmrmr.append(row[9])
                case 'f_classif':
                    Accf_classif.append(row[9])
                case 'RFE':
                    AccRFE.append(row[9])
                case 'ReliefF':
                    AccReliefF.append(row[9])
        # perform Friedman Test
        result = stats.friedmanchisquare(AccAHFS, AccFSS, Accmrmr,Accf_classif,AccRFE,AccReliefF)
        print(file,result)
    except:
        print(file, 'exeption')



# cwd = os.path.abspath('finished improved algo')
# filesNew= os.listdir(cwd)