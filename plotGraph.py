
import matplotlib.pyplot as plt

def plot_metrics(metric_values, epochs, labels, title, ylabel, xlabel="Epoch", linestyle=None):
    plt.figure(figsize=(12, 6))
    for i, values in enumerate(metric_values):
        plt.plot(epochs, values, label=labels[i], marker='o', linestyle=linestyle[i] if linestyle else '-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()