import copy
import os.path

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from evotorch.algorithms import SNES,CEM,PGPE,XNES
from evotorch.logging import PandasLogger, StdOutLogger
from evotorch.neuroevolution import SupervisedNE
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd


file= "metrics.csv"


def get_df(file,columns):
    df=None
    if os.path.exists(file):
        df = pd.read_csv(file, index_col=0)

    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(file)
    return df



def check_and_create(path):
    if not os.path.exists(path):
        os.mkdir(path)

metrics={
    "best_eval":-100000,
    "worst_eval":100000000,
    "iter":1,
    "median_eval":-100000,
    "pop_best_eval":-100000,
    "mean_eval":-10000
}



env_name = 'iris-data'

folder_name = f"./data/{env_name}"
check_and_create(folder_name)
weights_path = f"{folder_name}/weights"
check_and_create(weights_path)
save_weights_after_iter=1

def check_points():

    global searcher,metrics,problem,weights_path,save_weights_after_iter,check_and_create

    print(searcher.status)
    current_status=copy.deepcopy(searcher.status)

    if not "iter" in current_status:
        return
    iter=current_status['iter']
    if not iter >=save_weights_after_iter:
        return
    print("ended")

    def save_weights_based_on_metrics(metric_name, sol_name):
        if metric_name in current_status and current_status[metric_name] >= metrics[metric_name]:
            current_score = current_status[metric_name]


            solution = copy.deepcopy(searcher.status[sol_name])
            best_policy = problem.parameterize_net(solution.access_values())
            path=f"{weights_path}/{metric_name}"
            check_and_create(path)
            file_name=f"{path}/iter_{iter}_score_{current_score}.pth"
            torch.save(best_policy.state_dict(), file_name)
            metrics[metric_name] = current_score
            print(f"saved {file_name}")

    m_name = "pop_best_eval"
    s_name = "pop_best"
    save_weights_based_on_metrics( m_name, s_name)
    m_name = "best_eval"
    s_name = "best"
    save_weights_based_on_metrics( m_name, s_name)
    m_name = "median_eval"
    s_name = "pop_best"
    save_weights_based_on_metrics( m_name, s_name)
    m_name = "mean_eval"
    s_name = "pop_best"
    save_weights_based_on_metrics( m_name, s_name)

columns=["Id","epoch", "parameters",
                                   "population_size", "best_eval",
                                   "train_precision",
                                   "train_recall", "train_f1", "train_accuracy",
                                   "val_precision",
                                   "val_recall", "val_f1", "val_accuracy",
         "mean_eval","median_eval","pop_best_eval",
         "algorithm"
                                   ]
df=get_df(file,columns)
seed=42


def get_dataset():

    iris = load_iris()
    X, Y = iris.data, iris.target
    # Convert the data to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)  # Use long type for classification
    # Split the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=seed)
    # Create TensorDatasets for training and validation
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    return train_loader,val_loader,train_dataset


# Load the Iris dataset
train_loader,val_loader,train_dataset=get_dataset()


# Define the neural network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def interact(model, loader):
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for X_batch, Y_batch in loader:
            outputs = model(X_batch)
            _, predictions = torch.max(outputs, 1)
            y_true.extend(Y_batch.numpy())
            y_pred.extend(predictions.numpy())
    return y_pred, y_true
def compute_metrics(y_true, y_pred):
    """
    Compute precision, recall, f1-score, and accuracy for multi-class classification.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    dict: A dictionary containing precision, recall, f1-score, and accuracy.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1, accuracy
def cal_score():
    global searcher, val_loader, \
        train_loader, compute_metrics,\
        get_id_value,count_parameters,pop_size,algorithm
    bestsol = copy.deepcopy(searcher.status["pop_best"])
    epoch= copy.deepcopy(searcher.status["iter"])
    loss = copy.deepcopy(searcher.status["best"])
    best_model = problem.parameterize_net(bestsol.access_values())
    parameters_count=count_parameters(best_model)

    # Calculate metrics for the training set
    y_pred_train, y_true_train = interact(best_model, train_loader)
    train_precision, train_recall, train_f1, train_accuracy = compute_metrics(y_true_train, y_pred_train)
    train_metrics = {
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_accuracy": train_accuracy
    }

    # Calculate metrics for the validation set
    y_pred_val, y_true_val = interact(best_model, val_loader)
    val_precision, val_recall, val_f1, val_accuracy = compute_metrics(y_true_val, y_pred_val)
    val_metrics = {
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_accuracy": val_accuracy
    }
    id={"Id":get_id_value(epoch,parameters_count,pop_size,algorithm),
        "population_size":pop_size,"parameters":parameters_count,
        "loss":loss,
        "algorithm":algorithm,
        "epoch":epoch}


    # Combine train and validation metrics
    metrics = {**id,**train_metrics, **val_metrics}
    return metrics
pop_sizes = [ 50, 100]
neurons = [8, 16,32,64,128,256]
algorithms=["SNES","CEM",]
epochs = 100


def get_id_value(epochs, parameters, size,algorithm):
    return f"epoch:{epochs},parameters:{parameters},pop_size:{size},algorithm:{algorithm}"


pop_size=0
algorithm=""
print(f"{df.tail()}")


def get_searcher(problem,size,algorithm):
    if algorithm=="SNES":
        return SNES(problem, popsize=size, radius_init=2.25, )
    elif algorithm=="PGPE":
        return PGPE(problem, popsize=size, radius_init=2.25,center_learning_rate=0.001,stdev_learning_rate=0.5 )
    elif algorithm=="CEM":
        return CEM(problem, popsize=size, parenthood_ratio=0.2,radius_init=2.25, )
    elif algorithm=="XNES":
        return XNES(problem, popsize=size, radius_init=2.25, )


for pop_size in pop_sizes:
    for neuron in neurons:
        for algorithm in algorithms:

            network = nn.Sequential(
            nn.Linear(4, neuron),  # 4 input features for the Iris dataset
            nn.ReLU(),
            nn.Linear(neuron, 3),  # 3 output neurons for 3 classes
            )
            parameters=count_parameters(network)
            id=get_id_value(epochs,parameters,pop_size,algorithm)
            if ((df["Id"] == id) ).any():
                print(f"already done:{id}")
                continue

            problem = SupervisedNE(
            dataset=train_dataset,
            network=network,

            minibatch_size=32,
            loss_func=nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
            )
            searcher = get_searcher(problem,pop_size,algorithm)



            searcher.after_step_hook.append(cal_score)
            #searcher.before_step_hook.append(check_points)
            pandas_logger = PandasLogger(searcher)
            std_logger = StdOutLogger(searcher)
            searcher.run(epochs)
            new_df = pandas_logger.to_dataframe()
            df = pd.concat([df, new_df[columns]], ignore_index=True)
            print(f"done:{id}")
            df.to_csv(file)





