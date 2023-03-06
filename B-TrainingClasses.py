from URSGeneClassifier import *
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics, model_selection, svm, linear_model
from sklearn.utils import shuffle
from random import randrange
import numpy as np
import pickle
import os
import pandas
import matplotlib.pyplot as plt
import scipy.sparse as sps

def createTrainingClass(labels, output, id_dict, sp_matrix):
    positiveClass, set_pos = create_positive_class(labels, sp_matrix, id_dict)
    negativeClass, set_neg = create_negative_class(set_pos, sp_matrix, id_dict)
    fullTrain = create_full_training_class(positiveClass, negativeClass, output)
    # print("Positive class size: " + str(len(set_pos)) + ", Negative class size: " + str(len(set_neg)))
    
print("Restructuring and formatting GIANT network...")
id_dict = re_id_genes_to_dict("./data/GIANT_brain_top.txt", "./data/GIANT_brain_top_restruc.txt"
                           , "./data/saved_dictionary.pkl")

print("Creating sparse matrix from network...")
sp_matrix, dim = create_sparse_matrix(id_dict, "./data/GIANT_brain_top_restruc.txt")

print("Creating Alzheimer's training class...")
createTrainingClass('./data/alz_pos_class_labels.txt', './training/alz_full_training.pkl', id_dict, sp_matrix)

print("Creating Schizophrenia training class...")
createTrainingClass('./data/scz_pos_class_labels.txt', './training/scz_full_training.pkl', id_dict, sp_matrix)

print("Creating Bipolar training class...")
createTrainingClass('./data/bipolar_pos_class_labels.txt', 
                    './training/bipolar_full_training.pkl', id_dict, sp_matrix)

print("Creating Autism training class...")
createTrainingClass('./data/autism_pos_class_labels.txt', 
                    './training/autism_full_training.pkl', id_dict, sp_matrix)