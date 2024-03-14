import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp

##################################### Plot Confusion Matrix With Aggregates and % Display #################################

def calc_confusion_matrix(y_pred, y_true,
                           class_names=['Feet', 'Left Hand', 'Right Hand', 'Tongue'],
                           savefig=False, fig_path='../Gallery/confusion_matrix.png'):
    """
    Plot confusion matrix.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - class_names: List of class names
    - savefig: Whether to save the figure
    - fig_path: Path to save the figure
    Note: y_true and y_pred are either both one-hot encoded or categorical
    """
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix

    # Add aggregates for specificity and sensitivity
    specificity = np.diag(cm) / np.sum(cm, axis=1)
    sensitivity = np.diag(cm) / np.sum(cm, axis=0)

    # Extend class names with specificity and sensitivity labels
    extended_class_names = class_names + ['Specificity', 'Sensitivity']

    # Extend confusion matrix with specificity and sensitivity values
    cm_ext = np.zeros((num_classes + 2, num_classes + 2))
    cm_ext[:num_classes, :num_classes] = cm_normalized
    cm_ext[-2, :num_classes] = specificity
    cm_ext[:num_classes, -2] = sensitivity

    plt.imshow(cm_ext, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(num_classes + 2)
    plt.xticks(tick_marks, extended_class_names, rotation=45)
    plt.yticks(tick_marks, extended_class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(num_classes + 2):
        for j in range(num_classes + 2):
            plt.text(j, i, '{:.2%}'.format(cm_ext[i, j]), ha='center', va='center', color='red')

    # Save the figure
    if savefig:
        plt.savefig(fig_path)
    plt.show()


##################################### Plot Confusion Matrix #################################

def print_confusion_matrix(y_pred, y_true,
                          class_names=['Feet', 'Left Hand', 'Right Hand', 'Tongue'],
                          savefig=False, fig_path='../Gallery/confusion_matrix.png'):
    """
    Plot confusion matrix.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - class_names: List of class names
    - savefig: Whether to save the figure
    - fig_path: Path to save the figure
    Note: y_true and y_pred are either both one-hot encoded or categorical
    """
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    # Save the figure
    if savefig:
        plt.savefig(fig_path)
    plt.show()

##################################### Model Metrics #################################
    
def metrics_report(y_pred, y_true, matrix_on=False, categorical=True, savefig=False, fig_path='../Gallery/confusion_matrix.png'):
    """
    Generate and display classification metrics.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - matrix_on: Whether to display the confusion matrix
    - categorical: Whether the data is one-hot encoded
    - savefig: Whether to save the figure
    - fig_path: Path to save the figure
    """
    if not categorical:
        # Data is one-hot encoded, convert to categorical labels
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    # List of class names (adjust as per your dataset)
    class_names = ['Feet', 'Left Hand', 'Right Hand', 'Tongue']

    if matrix_on:
        # Plot and display the confusion matrix
        # Display the confusion matrix
        conf_mat = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=np.arange(4))
        disp.plot(cmap='viridis', values_format='d') # .2% for percent display option

    # Calculate and print Cohen's kappa and Matthews correlation coefficient
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc}")

    # Generate and display the classification report
    report = classification_report(y_true, y_pred, target_names=np.arange(4).astype(str))
    print(report)

    if savefig:
        plt.savefig(fig_path)
    plt.show()

##################################### ROC / AUC Curve #################################



def ROC(y_pred, y_true, class_names = ['Feet', 'Left Hand', 'Right Hand', 'Tongue'], 
        categorical=True, savefig=False, fig_path='../Gallery/ROC.png'):
    """
    Generate and display ROC / AUC curve

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - class_names: class labels
    - categorical: Whether the data is categorical (true) or one-hot encoded (false)
    - savefig: Whether to save the figure
    - fig_path: Path to save the figure
    """

    if not categorical:
        # 1-hot encoded -> categorical
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        n_classes = y_true.shape[1]  # Number of classes
    else:
        n_classes = np.max(y_true) + 1
    
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    #print("classes:", n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(10, 7))
    colors = cycle(['aqua', 'darkorange', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve {class_names[i]} (area = {roc_auc[i]:.2f})')
                #label='ROC curve (area = {:.2f})'.format(roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {:.2f})'.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')

    if savefig:
        plt.savefig(fig_path)
    plt.show()
