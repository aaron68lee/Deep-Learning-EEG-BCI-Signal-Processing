import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

##################################### Plot Confusion Matrix #################################

def confusion_matrix(y_true, y_pred,
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
    
def metrics_report(y_true, y_pred, matrix_on=True, categorical=True, savefig=False, fig_path='../Gallery/confusion_matrix.png'):
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
        y_true_categorical = np.argmax(y_true, axis=1)
        y_pred_categorical = np.argmax(y_pred, axis=1)

    # List of class names (adjust as per your dataset)
    class_names = ['Feet', 'Left Hand', 'Right Hand', 'Tongue']

    if matrix_on:
        # Plot and display the confusion matrix
        # Display the confusion matrix
        conf_mat = confusion_matrix(y_true_categorical, y_pred_categorical)
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=np.arange(4))
        disp.plot(cmap='viridis', values_format='d')

    # Generate and display the classification report
    report = classification_report(y_true_categorical, y_pred_categorical, target_names=np.arange(4).astype(str))
    print(report)

    if savefig:
        plt.savefig(fig_path)
    plt.show()