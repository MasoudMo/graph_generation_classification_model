import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_conf_mat(confusion_mat):
    disp = ConfusionMatrixDisplay(confusion_mat)
    disp.plot()

    plt.show()


if __name__ == "__main__":
    conf_mat = np.array([[52, 12], [14, 75]])
    plot_conf_mat(conf_mat)