import os
import datetime

def read_training_data(file_path, type):
    print(f"Reading {type}ing data from {file_path}")
    if type=="train":
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        training_data = []
        for line in lines:
            if line.startswith("Epoch"):
                epoch_data = {}
                line_data = line.split(", ")
                #print(line_data)
                epoch_data["Epoch"] = int(line_data[0].split(" ")[1].split("/")[0])
                epoch_data["Train Accuracy"] = float(line_data[2].split(": ")[1])
                epoch_data["Train AUC-ROC"] = float(line_data[3].split(": ")[1])
                epoch_data["Train F1 Score"] = float(line_data[4].split(": ")[1])
                epoch_data["Train Precision"] = float(line_data[5].split(": ")[1])
                epoch_data["Train Recall"] = float(line_data[6].split(": ")[1])
                training_data.append(epoch_data)
        
        return training_data
    else:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        training_data = []
        for line in lines:
            if line.startswith("Epoch"):
                epoch_data = {}
                line_data = line.split(",")
                #print(line_data)
                epoch_data["Epoch"] = int(line_data[0].split(" ")[1].split("/")[0])
                epoch_data["Test Accuracy"] = float(line_data[1].split(": ")[1])
                epoch_data["Test AUC-ROC"] = float(line_data[2].split(": ")[1])
                epoch_data["Test F1 Score"] = float(line_data[3].split(": ")[1])
                epoch_data["Test Precision"] = float(line_data[4].split(": ")[1])
                epoch_data["Test Recall"] = float(line_data[5].split(": ")[1])
                training_data.append(epoch_data)
        
        return training_data

def find_highest_metrics(training_data,type):
    if type=="train":
        highest_accuracy = max(training_data, key=lambda x: x["Train Accuracy"])
        highest_auc_roc = max(training_data, key=lambda x: x["Train AUC-ROC"])
        highest_f1_score = max(training_data, key=lambda x: x["Train F1 Score"])
        highest_precision = max(training_data, key=lambda x: x["Train Precision"])
        highest_recall = max(training_data, key=lambda x: x["Train Recall"])
        
        return highest_accuracy, highest_auc_roc, highest_f1_score, highest_precision, highest_recall
    else:
        highest_accuracy = max(training_data, key=lambda x: x["Test Accuracy"])
        highest_auc_roc = max(training_data, key=lambda x: x["Test AUC-ROC"])
        highest_f1_score = max(training_data, key=lambda x: x["Test F1 Score"])
        highest_precision = max(training_data, key=lambda x: x["Test Precision"])
        highest_recall = max(training_data, key=lambda x: x["Test Recall"])
        
        return highest_accuracy, highest_auc_roc, highest_f1_score, highest_precision, highest_recall

def generate_report(dataset_name, today_date, highest_accuracy, highest_auc_roc, highest_f1_score, highest_precision, highest_recall,type,file_path):
    if type=="train":
        report = f"Dataset: {dataset_name}\n"
        report +=f"Log path: {file_path}\n"
        report += f"Date: {today_date}\n\n"
        
        report += "Epoch with highest Train Accuracy:\n"
        report += str(highest_accuracy) + "\n\n"
        
        report += "Epoch with highest Train AUC-ROC:\n"
        report += str(highest_auc_roc) + "\n\n"
        
        report += "Epoch with highest Train F1 Score:\n"
        report += str(highest_f1_score) + "\n\n"
        
        report += "Epoch with highest Train Precision:\n"
        report += str(highest_precision) + "\n\n"
        
        report += "Epoch with highest Train Recall:\n"
        report += str(highest_recall) + "\n\n"
        
        return report
    else:
        report = f"Dataset: {dataset_name}\n"
        report += f"Date: {today_date}\n\n"
        
        report += "Epoch with highest Test Accuracy:\n"
        report += str(highest_accuracy) + "\n\n"
        
        report += "Epoch with highest Test AUC-ROC:\n"
        report += str(highest_auc_roc) + "\n\n"
        
        report += "Epoch with highest Test F1 Score:\n"
        report += str(highest_f1_score) + "\n\n"
        
        report += "Epoch with highest Test Precision:\n"
        report += str(highest_precision) + "\n\n"
        
        report += "Epoch with highest Test Recall:\n"
        report += str(highest_recall) + "\n\n"
        
        return report

def save_report(report, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    today_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{folder_path}/report_{today_date}.txt"
    with open(file_name, 'w') as file:
        file.write(report)
    print(f"Report saved successfully at {file_name}")

def main():
    file_path = "output/bace/test/test_accuracy_details_2024-05-01_16-04-26.txt"
    dataset_name = "bace"
    type = "test"
    loss="recon" # bin or recon
    today_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
    training_data = read_training_data(file_path,type)
    highest_accuracy, highest_auc_roc, highest_f1_score, highest_precision, highest_recall = find_highest_metrics(training_data,type)
    
    report = generate_report(dataset_name, today_date, highest_accuracy, highest_auc_roc, highest_f1_score, highest_precision, highest_recall,type,file_path)
    
    folder_path = f"Report/{dataset_name}/{type}/{loss}"
    save_report(report, folder_path)

if __name__ == "__main__":
    main()
