import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(weights, y_test, y_pred, metric_path, classes=None, figsize=(16, 16), text_size=16):
    '''
    y_test --> truth labels
    y_pred --> predicted labels

    classes --> number of classes / labels in your dataset (10 classes for this example)
    figsize --> (10 , 10) has been set as a default figsize. Can be adjusted for our needs.
    text_size --> size of the text
    '''

    # Setting the default figsize
    figsize = figsize
    # Create the confusion matrix from sklearn
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize our confusion matrix

    # Number of clases
    n_classes = cm.shape[0]

    # Making our plot pretty
    fig, ax = plt.subplots(figsize=figsize)
    # Drawing the matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set labels to be classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label axes
    ax.set(title='Confusion Matrix',
           xlabel='Predicted Label',
           ylabel='True Label',
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)
    # Set the xaxis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.tick_params(axis='both', which='major', labelsize=text_size)
    plt.xticks(rotation=45)

    # Adjust the label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]}\n ({cm_norm[i, j] * 100:.1f}%)',
                 horizontalalignment='center', verticalalignment='center',
                 color='white' if cm[i, j] > threshold else 'black',
                 size=text_size)

    plt.savefig(metric_path / f'{weights}_ConfusionMatrix.png')