import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    roc_curve
)


class PimpPlot(object):
    save = None
    img_folder = None

    def __init__(self, save = False, folder = "figures"):
        """Plotting object.
        """
        self.save = save
        self.img_folder = folder
        if self.save and not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

    def __repr__(self):
        if self.save:
            return "The plots will be saved in './{}'".format(self.img_folder)

    def plot_roc(self, true, pred, label):
        """Plot a roc curve.
        """
        fpr, tpr, _ = roc_curve(true, pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange",
                lw=lw, label="ROC curve (area = {})".format(np.round(roc_auc, 2)))
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC {}".format(label))
        plt.legend(loc="lower right")
        if self.save:
            plt.savefig(os.path.join(self.img_folder, "{}_ROC.png".format(label)))

    def plot_distributions(self, true, pred, label):
        """Plot binary distributions.
        """
        plt.figure()
        plt.xlim([-0.3, 1.1])
        plt.hist(pred[np.where(true == 1)], color="darkorange", label="Ones", density=True, alpha=0.7)
        plt.hist(pred[np.where(true == 0)], color="navy", label="Zeros", density=True, alpha=0.7)
        plt.title("Predictions distribution of {}".format(label))
        plt.legend(loc="upper left")
        if self.save:
            plt.savefig(os.path.join(self.img_folder, "{}_distributions.png".format(label)))

    def plot_confusion_matrix(self, true, pred, classes, label, normalize=True):
        """Plot a confusion matrix.
        """
        cmap = plt.cm.OrRd
        cm = confusion_matrix(true, pred)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title("Confusion matrix {}".format(label))
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.grid(False)
        if self.save:
            plt.savefig(os.path.join(self.img_folder, "{}_thresholds.png".format(label)))

    def find_threshold_max_f1(self, true, pred, label, N = 9):
        """Find the f1_score maximum and plot it.
        """
        if not N % 2:
            N = N + 1
        thresholds = []
        results = []
        for _i in range(N):
            thr = np.round((_i + 1) / (N + 1), 2)
            thresholds.append(thr)
            pred_good = np.where(pred >= thr, 1, 0)
            results.append(f1_score(true, pred_good))
        
        argbest = np.argmax(results)
        best = thresholds[argbest]
        
        plt.figure()
        plt.scatter(best, results[argbest], color = "darkorange", label="Threshold maximising F1 = {}".format(best))
        thresholds.remove(thresholds[argbest])
        results.remove(results[argbest])
        plt.scatter(thresholds, results, color = "navy")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.title("Scatter F1 vs Threshold for {}".format(label))
        plt.legend(loc="upper right")
        if self.save:
            plt.savefig(os.path.join(self.img_folder, "{}_confusion.png".format(label)))
        
        return best
