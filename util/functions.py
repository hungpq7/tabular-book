def plotPRChart(yTrue, yProb):
    from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = precision_recall_curve(yTrue, yProb)
    precision = precision[:-1]
    recall = recall[:-1]

    fig, ax = plt.subplots()
    ax.plot(threshold, precision)
    ax.plot(threshold, recall)
    ax.set_xlabel('Threshold')
    ax.legend(['Precision', 'Recall'], loc='best')
    ax.axis('scaled')
    plt.show()