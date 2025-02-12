import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Llama3.1 JSON model
llama_json_k0 = [(0.7653, 0.0181), (0.7975, 0.0215), (0.7808, 0.0122), (0.6965, 0.0111)]
llama_json_k2 = [(0.8508, 0.0259), (0.7000, 0.0410), (0.7676, 0.0303), (0.6885, 0.0273)]
llama_json_k4 = [(0.9042, 0.0227), (0.6788, 0.0432), (0.7746, 0.0278), (0.6965, 0.0341)]
llama_json_k6 = [(0.0137, 0.0227), (0.0120, 0.0057), (0.0126, 0.0051), (0.0107, 0.0061)]
llama_json_k8 = [(0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000)]
llama_json_k10 = [(0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000), (0.0000, 0.0000)]

# Llama3.1 Code style
llama_code1_k0 = [(0.5816, 0.0425), (0.6949, 0.0478), (0.6332, 0.0448), (0.5814, 0.0440)]
llama_code1_k2 = [(0.8518, 0.0231), (0.7468, 0.0176), (0.7958, 0.0189), (0.7413, 0.0185)]
llama_code1_k4 = [(0.8198, 0.0176), (0.6524, 0.0078), (0.7265, 0.0070), (0.6804, 0.0093)]
llama_code1_k6 = [(0.0781, 0.0354), (0.0650, 0.0350), (0.0704, 0.0352), (0.0603, 0.0286)]
llama_code1_k8 = [(0.0650, 0.0381), (0.0532, 0.0311), (0.0571, 0.0316), (0.0502, 0.0284)]
llama_code1_k10 = [(0.0474, 0.0175), (0.0384, 0.0225), (0.0404, 0.0169), (0.0343, 0.0128)]


# Gpt-4o-mini JSON model
gpt_json_k0 = [(0.8338, 0.0141), (0.8049, 0.0113), (0.8191, 0.0105), (0.7646, 0.0095)]
gpt_json_k2 = [(0.8814, 0.0147), (0.7379, 0.0128), (0.8032, 0.0101), (0.7383, 0.0121)]
gpt_json_k4 = [(0.9114, 0.0194), (0.7281, 0.0186), (0.8094, 0.0165), (0.7401, 0.0173)]
gpt_json_k6 = [(0.8983, 0.0139), (0.7123, 0.0138), (0.7945, 0.0110), (0.7144, 0.0132)]
gpt_json_k8 = [(0.9245, 0.0080), (0.7113, 0.0102), (0.8040, 0.0081), (0.7276, 0.0139)]
gpt_json_k10 = [(0.9174, 0.0097), (0.7113, 0.0102), (0.8013, 0.0088), (0.7368, 0.0160)]

# Gpt-4o-mini Code style
gpt_code1_k0 = [(0.7584, 0.0197), (0.8818, 0.0092), (0.8152, 0.0085), (0.7636, 0.0115)]
gpt_code1_k2 = [(0.8731, 0.0134), (0.7793, 0.0176), (0.8235, 0.0146), (0.7583, 0.0153)]
gpt_code1_k4 = [(0.8971, 0.0131), (0.7468, 0.0133), (0.8150, 0.0117), (0.7455, 0.0193)]
gpt_code1_k6 = [(0.9037, 0.0118), (0.7300, 0.0118), (0.8076, 0.0113), (0.7387, 0.0055)]
gpt_code1_k8 = [(0.8905, 0.0175), (0.7448, 0.0172), (0.8111, 0.0168), (0.7420, 0.0170)]
gpt_code1_k10 = [(0.9071, 0.0094), (0.7399, 0.0041), (0.8150, 0.0054), (0.7491, 0.0078)]


# Data preparation
ks = [0, 2, 4, 6, 8, 10]

metrics = {
    'LLaMa3.1 JSON-style': [llama_json_k0, llama_json_k2, llama_json_k4, llama_json_k6, llama_json_k8, llama_json_k10],
    'LLaMa3.1 Code-style': [llama_code1_k0, llama_code1_k2, llama_code1_k4, llama_code1_k6, llama_code1_k8, llama_code1_k10],
    'GPT-4o-mini JSON-style': [gpt_json_k0, gpt_json_k2, gpt_json_k4, gpt_json_k6, gpt_json_k8, gpt_json_k10],
    'GPT-4o-mini Code-style': [gpt_code1_k0, gpt_code1_k2, gpt_code1_k4, gpt_code1_k6, gpt_code1_k8, gpt_code1_k10],
}

# Colors for the lines based on model/style
colors = {
    'LLaMa3.1 JSON-style': '#ff9933',
    'LLaMa3.1 Code-style': '#b35900',
    'GPT-4o-mini JSON-style': '#1a8cff',
    'GPT-4o-mini Code-style': '#0059b3',
}

# Metric names
metric_names = ['Precision', 'Recall', 'F1', 'Jaccard']

for i, metric_name in enumerate(metric_names):
    plt.figure(figsize=(8, 4))  # New figure for each metric
    for model, data in metrics.items():
        means = [d[i][0] for d in data]
        stds = [d[i][1] for d in data]
        plt.errorbar(ks, means, yerr=stds, label=model, fmt='-o', capsize=5, color=colors[model])
    # plt.title(f'{metric_name} by k', fontsize=16)
    plt.xlabel('k', fontsize=16)
    # plt.ylabel(metric_name, fontsize=16)
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=16)
    plt.xticks(ks, fontsize=16)
    plt.grid(True)
    if "Jaccard" in metric_name:
        plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/effect_k_{}.png".format(metric_name))

