import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the tables for Llama3, Llama3.1, Gpt4o-mini
methods = ['zero-shot', 'few-shot', 'cot', 'two stage']

# Data for Precision Micro (mean and std deviation)
precision_micro_data = {
    'Llama3': [(0.8282, 0.0215), (0.9009, 0.0084), (0.8180, 0.0249), (0.8415, 0.0290)],
    'Llama3.1': [(0.7773, 0.0200), (0.8813, 0.0205), (0.7566, 0.0223), (0.8302, 0.0248)],
    'Gpt4o-mini': [(0.8199, 0.0124), (0.8908, 0.0148), (0.7995, 0.0209), (0.8776, 0.0202)]
}

# Data for Recall Micro (mean and std deviation)
recall_micro_data = {
    'Llama3': [(0.8000, 0.0200), (0.6941, 0.0200), (0.8015, 0.0200), (0.7034, 0.0248)],
    'Llama3.1': [(0.8207, 0.0321), (0.6952, 0.0362), (0.8402, 0.0126), (0.7399, 0.0366)],
    'Gpt4o-mini': [(0.8020, 0.0157), (0.7416, 0.0041), (0.8317, 0.0201), (0.7446, 0.0151)]
}

# Data for F1 Micro (mean and std deviation)
f1_micro_data = {
    'Llama3': [(0.8137, 0.0172), (0.7840, 0.0122), (0.8095, 0.0191), (0.7659, 0.0195)],
    'Llama3.1': [(0.7984, 0.0249), (0.7764, 0.0178), (0.7961, 0.0147), (0.7821, 0.0272)],
    'Gpt4o-mini': [(0.8108, 0.0122), (0.8093, 0.0052), (0.8152, 0.0166), (0.8056, 0.0163)]
}

# Function to plot Precision, Recall, F1 Micro for all models
def plot_scores(data, title):
    models = list(data.keys())
    x = np.arange(len(methods))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        means = [score[0] for score in data[model]]
        errors = [score[1] for score in data[model]]
        ax.bar(x + i*width, means, width, yerr=errors, label=model)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.legend()

    plt.show()

# Plotting Precision Micro, Recall Micro, F1 Micro
plot_scores(precision_micro_data, 'Precision Micro')
plot_scores(recall_micro_data, 'Recall Micro')
plot_scores(f1_micro_data, 'F1 Micro')
