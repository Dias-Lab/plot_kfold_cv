import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from sklearn.multioutput import MultiOutputClassifier
from itertools import cycle
from sklearn import svm
import numpy as np

def run_cv_and_plot_auc(name, classifier,cv, X, y):

    interp_mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    fig.set_size_inches(7, 6)

    # Binarize the output
    yn = label_binarize(y, classes=np.unique(y))

    if len(y[:,0].shape)<=1:
        n_classes = 1
    else:
        n_classes = yn.shape[1]
    
    my_split = cv.split(X, y)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
   
    for i in range(n_classes):
        tpr[i] = []
        fpr[i] = []
        roc_auc[i] = []

    for k, (train, test) in enumerate(my_split):
        #print("yn shape",yn[train].shape, np.sum(yn[train],0))
        if n_classes == 1:
            classifier.fit(X[train], np.ravel(yn[train]))
        else:
            classifier.fit(X[train], yn[train])
            
        y_score = classifier.predict_proba(X[test])
        
        #https://stackoverflow.com/questions/67104496/how-to-correctly-reshape-the-multiclass-output-of-predict-proba-of-a-sklearn-cla
        if n_classes > 1:
            y_score = np.asarray(y_score)[:,:, 1].T
        else:
            y_score = np.asarray(y_score[:,1]).reshape([-1,1])
        #https://localcoder.org/computing-scikit-learn-multiclass-roc-curve-with-cross-validation-cv
        for i in range(n_classes):

            fpr_tmp, tpr_tmp, _ = roc_curve(yn[test, i], y_score[:, i])
            interp_tpr = np.interp(interp_mean_fpr, fpr_tmp, tpr_tmp)
            interp_tpr[0] = 0.0
            fpr[i].append(list(fpr_tmp))
            tpr[i].append(list(interp_tpr))
            auc_score_tmp = auc(fpr_tmp,tpr_tmp)
            roc_auc[i].append(auc_score_tmp)
            
            print("k =", k+1, "AUC score:", auc_score_tmp)
            print("k =", k+1, "Mean FPR (Specificity):", np.mean(fpr_tmp), "std. dev.:", np.std(fpr_tmp))
            print("k =", k+1, "Mean TPR (Sensitivity):", np.mean(tpr_tmp), "std. dev.:", np.std(tpr_tmp))

    colors = cycle(['blue', 'green', 'red','purple'])

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance", alpha=0.8)

    for i, color in zip(range(n_classes), colors):

        mean_tpr = np.mean(tpr[i], axis=0)
        #mean_fpr = np.mean(fpr[i], axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = auc(interp_mean_fpr, mean_tpr)
        std_auc = np.std(roc_auc[i])

        ax.plot(
            interp_mean_fpr,
            mean_tpr,
            color=color,
            label=r"Class %0d: mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (i+1, mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tpr[i], axis=0)
        #std_fpr = np.std(fpr[i], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        ax.fill_between(
            interp_mean_fpr,
            tprs_lower,
            tprs_upper,
            color=color,
            alpha=0.2,
            label='_Hidden',
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=name,
        )
        ax.legend(loc="lower right")
        ax.set_xlabel('False Positive Rate (Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)') 
        print("Mean AUC score across all k folds:", mean_auc, "std. dev.:", std_auc)
    plt.show()