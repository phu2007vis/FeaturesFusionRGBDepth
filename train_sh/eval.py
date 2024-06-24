import sklearn.metrics as metrics
import os
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
class Eval:
    def __init__(self,run_name):
        #create a folder "dd-hh-mm-ss_[run_name]" in the current directory
        save_folder = "results"+os.path.sep +str(run_name)+os.path.sep+ datetime.datetime.now().strftime("%d-%H-%M-%S") 
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder
        os.makedirs(os.path.join(save_folder, "raw"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "confusion_matrix"), exist_ok=True)
        #create a csv file in the folder to save loss
        self.train_loss_file = os.path.join(save_folder, "train_loss.csv")
        self.valid_loss_file = os.path.join(save_folder,'valid_loss.csv')
        self.train_loss_file_handler = open(self.train_loss_file, "w")
        self.valid_loss_file_handler = open(self.valid_loss_file, "w")
        self.train_count_line = 0
        self.valid_count_line =  0
        self.add_train_loss("iter,train_loss")
        self.add_valid_loss("iter,valid_loss")
        
    def set_labels(self,labels):
        self.labels = labels
    def add_train_loss(self,entry):
        train_count_line = self.train_count_line
        self.train_loss_file_handler.write(f"{train_count_line},{entry}\n")
        self.train_count_line+=1
        #flush the buffer to write to the file
        self.train_loss_file_handler.flush()
    def add_valid_loss(self,entry):
        valid_count_line = self.valid_count_line
        self.valid_loss_file_handler.write(f"{valid_count_line},{entry}\n")
        self.valid_count_line+=1
        self.valid_loss_file_handler.flush()
    def evaluate(self,phase,ep,result,class_labels):
        raw_file = os.path.join(self.save_folder, "raw", f"{phase}_{ep}_raw.pkl")
        with open(raw_file, "wb") as f:
            pickle.dump(result, f)
        result_file = os.path.join(self.save_folder, f"{phase}_result.csv")
        #combine [Precision,Recall,F1 Score] for each class to make a list of [Precision_classname1,Recall_classname1,F1_classname1,Precision_classname2,Recall_classname2,F1_classname2,...]
        met_str = ["Precision","Recall","F1 Score"]
        comb_str = [f"{x}_{y}" for y in class_labels for x in met_str]
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                f.write(f"Accuracy,{','.join(comb_str)}\n")
                f.write(f"{self.calculate_metrics(result)}\n")
        else:
            with open(result_file, "a") as f:
                f.write(f"{self.calculate_metrics(result)}\n")

    def get_path(self):
        return self.save_folder
    def save_confusion_matrix(self, phase,ep, result,class_names,normalize = 'pred'):
        num_classes = len(class_names)

        predicted = [x[0] for x in result]
        labels = [x[1] for x in result]

        # Create a confusion matrix
        matrix = metrics.confusion_matrix(labels, predicted, normalize=normalize)

    
        display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=class_names)
        fig, ax = plt.subplots(figsize=(30,30))
        display.plot(ax=ax,cmap='Blues', values_format='.2f' if normalize else 'd')
        # Remove the values in the confusion matrix
        for text in display.text_.ravel():
            text.set_visible(False)

        # Customize plot titles and labels
        plt.title(f'Confusion Matrix - {phase.capitalize()} Phase - Epoch {ep}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=90)
        # Save the confusion matrix plot
        cm_folder = os.path.join(self.save_folder, "confusion_matrix")
        os.makedirs(cm_folder, exist_ok=True)
        cm_file = os.path.join(cm_folder, f"{phase}_{ep}_confusion_matrix.svg")
        plt.savefig(cm_file, dpi=300, format='svg', bbox_inches='tight')  # Increase dpi for higher resolution
        plt.close()


    def calculate_metrics(self,result):
        return_str = ""
        #calculate all the metrics, save as dec 4
        predicted = [x[0] for x in result]
        labels = [x[1] for x in result]
       
        unique_labels = np.unique(labels)
        unique_predicted = np.unique(predicted)

        if len(unique_labels) != len(unique_predicted):
            # Create a mapping dictionary to renumber the labels
            mapping = {label: i for i, label in enumerate(unique_labels)}
            
            # Update the labels and predicted using the mapping dictionary
            labels = [mapping[label] for label in labels]
            predicted = [mapping[label] for label in predicted]


        acc = metrics.accuracy_score(labels,predicted)
        return_str += f"{acc:.4f},"
        prec = metrics.precision_score(labels,predicted,average=None, zero_division = 0)
        rec = metrics.recall_score(labels,predicted,average=None,zero_division = 0 )
        f1 = metrics.f1_score(labels,predicted,average=None,zero_division = 0 )
        for i in range(len(prec)):
            return_str += f"{prec[i]:.4f},{rec[i]:.4f},{f1[i]:.4f},"
        return return_str
        



