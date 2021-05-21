import numpy as np
from sklearn.metrics import confusion_matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def accuracy(predictions, ground_truth):
    cm = confusion_matrix(ground_truth, predictions, labels=[0,1])
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    
    acc = (tp+tn)/(tn+fp+tp+fn)
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    
    return acc, tpr, tnr
                        
        

class Metrics():
    def __init__(self):
        self.cm = np.zeros((2,2))

    def update(self, predictions, ground_truth):
        self.cm += confusion_matrix(ground_truth, predictions, labels=[0,1])

    def compute_stats(self):
        tn = self.cm[0][0]
        fp = self.cm[0][1]
        fn = self.cm[1][0]
        tp = self.cm[1][1]

        acc = (tp+tn)/(tn+fp+tp+fn)
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)
        ppv = tp/(tp+fp)
        npv = tn/(tn+fn)

        # Reset
        self.cm = np.zeros((2,2))

        return acc, tpr, tnr, ppv, npv

    
def evaluate_similarity_prediction(model, data_loader, device):
    tn = 0
    fp = 0
    fn = 0
    tp = 0

    # Confusion matrix
    with torch.no_grad():
        for batch in data_loader:
            labels = batch['T'].to(device)
            predictions = model.forward((batch['X'].to(device), batch['B'].to(device))).argmax(-1)
            
            cm = confusion_matrix(labels.cpu(), predictions.cpu(), labels=[0,1])
            
            tn += cm[0][0]
            fp += cm[0][1]
            fn += cm[1][0]
            tp += cm[1][1]

    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)

    # Accuracy
    accuracy = []
    with torch.no_grad():
        for batch in dl:
            labels = batch['T'].to(device)
            predictions = model.forward((batch['X'].to(device), batch['B'].to(device))).argmax(-1)
            n_correct = (predictions == labels).sum().cpu().numpy()
            accuracy.append(n_correct/len(predictions))

    accuracy = sum(accuracy)/len(accuracy)


    return accuracy, tpr, tnr, ppv, npv
