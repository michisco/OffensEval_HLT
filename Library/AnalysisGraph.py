from sklearn.metrics import matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import warnings

def show_confusion_matrix(y_test, y_pred, dl):
    '''Draw a confusion matrix'''
    color = 'white'
    y_test_cpu = y_test.detach().cpu().clone().numpy()
    y_pred_cpu = y_pred.detach().cpu().clone().numpy()
    c_mat = confusion_matrix(y_test_cpu, y_pred_cpu)
    matrix = ConfusionMatrixDisplay(c_mat, display_labels=dl)
    matrix.plot()
    plt.show()

def show_confusion_matrixNoTorch(y_test, y_pred, dl):
    '''Draw a confusion matrix that not use torch'''
    color = 'white'
    c_mat = confusion_matrix(y_test, y_pred)
    matrix = ConfusionMatrixDisplay(c_mat, display_labels=dl)
    matrix.plot()
    plt.show()

def show_reportNoTorch(y_test, y_preds, tn):
    print(classification_report(y_test, y_preds, target_names=tn, digits = 4))
    
def show_report(y_test, y_preds, tn):
    y_test_cpu = y_test.detach().cpu().clone().numpy()
    y_pred_cpu = y_preds.detach().cpu().clone().numpy()
    print(classification_report(y_test_cpu, y_pred_cpu, target_names=tn, digits = 4))

def plot_curves_history(history, measurement):
     plt.plot(history.history[measurement])
     plt.plot(history.history['val_'+measurement], '')
     
     plt.xlabel("Epochs")
     plt.ylabel(measurement)
     plt.legend(["Training", "Validation"])
     plt.show()

def plot_curves(losses, accuracies, n_model, epochs):
    fig = plt.figure()
    
    ax0 = fig.add_subplot(121, title="Loss model %d" % (n_model))
    ax1 = fig.add_subplot(122, title="Accuracy model %d" % (n_model))
    
    ax0.plot(epochs, losses['train'], 'bo-', label='train')
    ax0.plot(epochs, losses['val'], 'ro-', label='val')
    ax1.plot(epochs, accuracies['train'], 'bo-', label='train')
    ax1.plot(epochs, accuracies['val'], 'ro-', label='val')
    plt.show()

def showWrongPredictions(df_test, test_labels, pred_labels):
    '''Show 5 wrong predictions from testset'''
    count = 0
    index_pos = 0
    
    for pred, label in zip(pred_labels, test_labels):
        if pred != label and label == 0:
            print(df_test.iloc[[index_pos]])
            count = count + 1
        
        if count == 5:
            break
        index_pos = index_pos + 1
    
    count = 0
    index_pos = 0
    
    for pred, label in zip(pred_labels, test_labels):
        if pred != label and label == 1:
            print(df_test.iloc[[index_pos]])
            count = count + 1
        
        if count == 5:
            break
        index_pos = index_pos + 1
    
    