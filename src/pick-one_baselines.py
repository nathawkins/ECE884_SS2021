import os
import json
import numpy as np
from argparse import ArgumentParser

from sklearn.model_selection import KFold

## Models from sklearn classification comparison benchmarking
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

## Performance metrics
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score

def load_embedding_from_file(fname):
    return np.load(fname)

def load_labels(fname):
    return np.load(fname)

def get_embedding_fname_list(data_dir, body = False, title = False):
    if body:
        return [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if "bodies" in fname]
    if title:
        return [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if "title" in fname]
    return [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]


def get_embedding_type_dict(title_embedding_files, bodies_embedding_files):
    embedding_types = {}
    for name in title_embedding_files:
        embedding_type = name.split("_")[-1].split(".")[0]
        try:
            embedding_types[embedding_type].append(name)
        except:
            embedding_types[embedding_type] = [name]
    for name in bodies_embedding_files:
        embedding_type = name.split("_")[-1].split(".")[0]
        try:
            embedding_types[embedding_type].append(name)
        except:
            embedding_types[embedding_type] = [name]
    return embedding_types

def load_embedding(fname):
    return np.load(fname)

def create_concatenated_features(title_X_mat, body_X_mat, title = False, body = False):
    if title:
        return title_X_mat
    if body:
        return body_X_mat

if __name__ == "__main__":
    ## Command line input for job index
    parser = ArgumentParser()
    parser.add_argument("-i", type = int, help = "Job array index")
    parser.add_argument("-title", action = "store_true", default = False)
    parser.add_argument("-body", action = "store_true", default = False)
    args = parser.parse_args()

    ## Define classifiers and name
    classifiers = [LogisticRegression(penalty = "l1", solver = "liblinear"),
                   LogisticRegression(penalty = "l2"),
                   KNeighborsClassifier(3),
                   KNeighborsClassifier(5),
                   KNeighborsClassifier(10),
                   SVC(kernel="linear"),
                   SVC(),
                   MLPClassifier(max_iter=1000),
                   GaussianNB(),
                   QuadraticDiscriminantAnalysis()]

    names = ["L1 Log Reg", "L2 Log Reg", "KNN-3", "KNN-5", "KNN-10","Linear SVM", "RBF SVM", "MLP", "Naive Bayes", "QDA"]

    ## Get list of filenames for titles and bodies
    title_embedding_files  = get_embedding_fname_list("../data", title = True)
    bodies_embedding_files = get_embedding_fname_list("../data", body = True)

    ## Get list of unique kinds of embeddings
    embedding_type_fnames = get_embedding_type_dict(title_embedding_files, bodies_embedding_files)

    ## Load labels
    y = load_labels("../data/labels.npy")

    ## Results to write to file
    results = {}

    ## Select feature set from job array index
    # for feature_set in embedding_type_fnames.keys():
    feature_set = list(embedding_type_fnames.keys())[args.i]
    print(feature_set)
    title_fname = embedding_type_fnames[feature_set][0]
    body_fname  = embedding_type_fnames[feature_set][1]
    X = create_concatenated_features(load_embedding(title_fname), load_embedding(body_fname), title = args.title, body = args.body)
    
    ## Loop over models
    for model_, model_name_ in zip(classifiers, names):
        print(model_name_)
        ## Make key for results
        results[feature_set+"_"+model_name_] = {"accuracy": 0.0, "auprc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.0}

        ## 10-fold CV split
        kf = KFold(n_splits = 10, random_state = 8675309, shuffle = True)
        i = 0
        for train_index, test_index in kf.split(X):
            i += 1
            print(i)
            ## Perform training and testing split
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            ## Train
            model_.fit(X_train, y_train)

            ## Predict
            y_pred = model_.predict(X_test)

            ## Score
            for metric, metric_name in zip([accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score],["accuracy", "auprc", "precision", "recall", "f1", "auroc"]):
                results[feature_set+"_"+model_name_][metric_name] += metric(y_test, y_pred)

        ## Average metric
        for k in results[feature_set+"_"+model_name_].keys():
            results[feature_set+"_"+model_name_][k] /= 10            

        print(results)
        if args.title:
            json.dump(results, open(f"../results/title_{args.i}.json", "w"))
        if args.body:
            json.dump(results, open(f"../results/body_{args.i}.json", "w"))
    
