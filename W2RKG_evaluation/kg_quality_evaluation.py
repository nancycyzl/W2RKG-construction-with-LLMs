import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


kg_eval_path = "W2RKG_evaluation/kg_test_dataset_combined.xlsx"
kg_eval_df = pd.read_excel(kg_eval_path)
print(kg_eval_df.shape)


# ------ each evaluator results plot and average ------
R1_waste_aligned = sum(kg_eval_df["R1_waste_aligned"].tolist())
R1_waste_valid = sum(kg_eval_df["R1_waste_valid"].tolist())
R1_waste_valid_pct = round(R1_waste_valid / R1_waste_aligned, 3)    
R1_resource_aligned = sum(kg_eval_df["R1_resource_aligned"].tolist())
R1_resource_valid = sum(kg_eval_df["R1_resource_valid"].tolist())
R1_resource_valid_pct = round(R1_resource_valid / R1_resource_aligned, 3)

R2_waste_aligned = sum(kg_eval_df["R2_waste_aligned"].tolist())
R2_waste_valid = sum(kg_eval_df["R2_waste_valid"].tolist())
R2_waste_valid_pct = round(R2_waste_valid / R2_waste_aligned, 3)    
R2_resource_aligned = sum(kg_eval_df["R2_resource_aligned"].tolist())
R2_resource_valid = sum(kg_eval_df["R2_resource_valid"].tolist())
R2_resource_valid_pct = round(R2_resource_valid / R2_resource_aligned, 3)

R3_waste_aligned = sum(kg_eval_df["R3_waste_aligned"].tolist())
R3_waste_valid = sum(kg_eval_df["R3_waste_valid"].tolist())
R3_waste_valid_pct = round(R3_waste_valid / R3_waste_aligned, 3)    
R3_resource_aligned = sum(kg_eval_df["R3_resource_aligned"].tolist())
R3_resource_valid = sum(kg_eval_df["R3_resource_valid"].tolist())
R3_resource_valid_pct = round(R3_resource_valid / R3_resource_aligned, 3)


print("Reviewer id: waste_aligned, waste_valid, waste_valid_pct, resource_aligned, resource_valid, resource_valid_pct")
print("R1: ", R1_waste_aligned, R1_waste_valid, R1_waste_valid_pct, R1_resource_aligned, R1_resource_valid, R1_resource_valid_pct)
print("R2: ", R2_waste_aligned, R2_waste_valid, R2_waste_valid_pct, R2_resource_aligned, R2_resource_valid, R2_resource_valid_pct)
print("R3: ", R3_waste_aligned, R3_waste_valid, R3_waste_valid_pct, R3_resource_aligned, R3_resource_valid, R3_resource_valid_pct)


waste_aligned_list = [R1_waste_aligned, R2_waste_aligned, R3_waste_aligned]
waste_valid_list = [R1_waste_valid, R2_waste_valid, R3_waste_valid]
waste_valid_pct_list = [R1_waste_valid_pct, R2_waste_valid_pct, R3_waste_valid_pct]
resource_aligned_list = [R1_resource_aligned, R2_resource_aligned, R3_resource_aligned]
resource_valid_list = [R1_resource_valid, R2_resource_valid, R3_resource_valid]
resource_valid_pct_list = [R1_resource_valid_pct, R2_resource_valid_pct, R3_resource_valid_pct]

# Calculate means
waste_aligned_mean = sum(waste_aligned_list) / len(waste_aligned_list)
waste_valid_mean = sum(waste_valid_list) / len(waste_valid_list)
waste_valid_pct_mean = sum(waste_valid_pct_list) / len(waste_valid_pct_list)
resource_aligned_mean = sum(resource_aligned_list) / len(resource_aligned_list)
resource_valid_mean = sum(resource_valid_list) / len(resource_valid_list)
resource_valid_pct_mean = sum(resource_valid_pct_list) / len(resource_valid_pct_list)

# Calculate standard deviations
waste_aligned_std = (sum((x - waste_aligned_mean) ** 2 for x in waste_aligned_list) / len(waste_aligned_list)) ** 0.5
waste_valid_std = (sum((x - waste_valid_mean) ** 2 for x in waste_valid_list) / len(waste_valid_list)) ** 0.5
waste_valid_pct_std = (sum((x - waste_valid_pct_mean) ** 2 for x in waste_valid_pct_list) / len(waste_valid_pct_list)) ** 0.5
resource_aligned_std = (sum((x - resource_aligned_mean) ** 2 for x in resource_aligned_list) / len(resource_aligned_list)) ** 0.5
resource_valid_std = (sum((x - resource_valid_mean) ** 2 for x in resource_valid_list) / len(resource_valid_list)) ** 0.5
resource_valid_pct_std = (sum((x - resource_valid_pct_mean) ** 2 for x in resource_valid_pct_list) / len(resource_valid_pct_list)) ** 0.5

print("\nAveraged:")
print("Waste aligned:", f"{waste_aligned_mean:.3f}±{waste_aligned_std:.3f}")
print("Waste valid:", f"{waste_valid_mean:.3f}±{waste_valid_std:.3f}")
print("Waste valid pct:", f"{waste_valid_pct_mean:.3f}±{waste_valid_pct_std:.3f}")
print("Resource aligned:", f"{resource_aligned_mean:.3f}±{resource_aligned_std:.3f}")
print("Resource valid:", f"{resource_valid_mean:.3f}±{resource_valid_std:.3f}")
print("Resource valid pct:", f"{resource_valid_pct_mean:.3f}±{resource_valid_pct_std:.3f}")



####  Plot the metrics
plt.rcParams.update({'font.size': 16})

# Define colors for evaluators
waste_colors = ['#ffbf80', '#ff8000', '#b35900']  # light, medium, dark orange
resource_colors = ['#BFD4E6', '#7FA7C9', '#3A75A9']  # light, medium, dark blue

# Figure 1: Waste Metrics
fig_waste = plt.figure(figsize=(8, 6))
ax_waste = fig_waste.add_subplot(111)

x = np.arange(2)  # Two metrics for waste
width = 0.25

# Prepare waste data
waste_metrics = [
    [R1_waste_aligned, R1_waste_valid],
    [R2_waste_aligned, R2_waste_valid],
    [R3_waste_aligned, R3_waste_valid]
]

# Plot waste metrics
for i, evaluator_data in enumerate(waste_metrics):
    ax_waste.bar(x + (i-1)*width, evaluator_data, width, 
                label=f'Evaluator {i+1}', 
                color=waste_colors[i])

# Customize waste plot
ax_waste.set_ylabel('Score (%)')
ax_waste.set_xticks(x)
ax_waste.set_xticklabels(['Alignment', 'Validness'])
# ax_waste.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)

# Add grid for waste plot
ax_waste.yaxis.grid(True, linestyle='--', alpha=0.7)
ax_waste.set_axisbelow(True)

# Save waste plot
plt.tight_layout()
plt.savefig('W2RKG_evaluation/kg_test_result_waste.png', 
            bbox_inches='tight', dpi=300)
plt.close()

# Figure 2: Resource Metrics
fig_resource = plt.figure(figsize=(8, 6))
ax_resource = fig_resource.add_subplot(111)

# Prepare resource data
resource_metrics = [
    [R1_resource_aligned, R1_resource_valid],
    [R2_resource_aligned, R2_resource_valid],
    [R3_resource_aligned, R3_resource_valid]
]

# Plot resource metrics
for i, evaluator_data in enumerate(resource_metrics):
    ax_resource.bar(x + (i-1)*width, evaluator_data, width, 
                   label=f'Evaluator {i+1}', 
                   color=resource_colors[i])

# Customize resource plot
ax_resource.set_ylabel('Score (%)')
ax_resource.set_xticks(x)
ax_resource.set_xticklabels(['Alignment', 'Validness'])
ax_resource.legend(loc='lower right')

# Add grid for resource plot
ax_resource.yaxis.grid(True, linestyle='--', alpha=0.7)
ax_resource.set_axisbelow(True)

# Save resource plot
plt.tight_layout()
plt.savefig('W2RKG_evaluation/kg_test_result_resource.png', 
            bbox_inches='tight', dpi=300)
plt.close('all')



# ------ internal agreement ------
def fleiss_kappa_three(r1, r2, r3, labels=None):
    # ---------- 0.  Input checks ----------
    r1, r2, r3 = map(np.asarray, (r1, r2, r3))
    if not (r1.shape == r2.shape == r3.shape):
        raise ValueError("All three rating arrays must have the same length.")

    N = len(r1)          # number of items
    n = 3                # ratings per item (fixed here)

    # ---------- 1.  Determine the label set ----------
    if labels is None:
        labels = np.union1d(np.union1d(r1, r2), r3)
    labels   = np.asarray(labels)
    k        = len(labels)
    lab2idx  = {lab: i for i, lab in enumerate(labels)}

    # ---------- 2.  Build the N×k count matrix  ✱ ----------
    count_mat = np.zeros((N, k), dtype=int)
    for i, (a, b, c) in enumerate(zip(r1, r2, r3)):
        # Fast bincount for the three ratings of this item
        count_mat[i] = np.bincount([lab2idx[a], lab2idx[b], lab2idx[c]],
                                   minlength=k)

    # ---------- 3.  Per-item observed agreement P_i  ✱ ----------
    # Numerator: sum_j n_ij (n_ij − 1)
    numer = (count_mat * (count_mat - 1)).sum(axis=1)
    P_i   = numer / (n * (n - 1))

    # ---------- 4.  Overall observed agreement ----------
    P_bar = P_i.mean()

    # ---------- 5.  Marginal proportions p_j  ✱ ----------
    p_j = count_mat.sum(axis=0) / (N * n)

    # ---------- 6.  Chance agreement ----------
    P_e_bar = np.square(p_j).sum()

    # ---------- 7.  Fleiss’ κ ----------
    if np.isclose(1.0 - P_e_bar, 0.0):
        raise ZeroDivisionError("Division by zero: expected agreement is 1.")
    kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
    return kappa


def cohen_kappa_score(rater1, rater2):
    if len(rater1) != len(rater2):
        raise ValueError("Input lists must have the same length.")

    # Get the set of all labels
    labels = set(rater1) | set(rater2)

    # Initialize confusion matrix
    confusion = {label: {l: 0 for l in labels} for label in labels}

    # Fill confusion matrix
    for a, b in zip(rater1, rater2):
        confusion[a][b] += 1

    # Convert confusion matrix to observed matrix and compute totals
    n = len(rater1)
    observed_agreement = sum(confusion[label][label] for label in labels) / n

    # Compute expected agreement
    rater1_counts = {label: sum(confusion[label].values()) for label in labels}
    rater2_counts = {label: sum(confusion[row][label] for row in labels) for label in labels}
    expected_agreement = sum(
        (rater1_counts[label] / n) * (rater2_counts[label] / n)
        for label in labels
    )

    # Compute kappa
    if expected_agreement == 1:
        return 1.0  # Avoid division by zero
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    print(f"Observed agreement: {observed_agreement}, Expected agreement: {expected_agreement}, Kappa: {kappa}")
    return kappa

def compute_kappa_score(rate1, rate2, rate3):
    kappa12 = cohen_kappa_score(rate1, rate2)
    kappa13 = cohen_kappa_score(rate1, rate3)
    kappa23 = cohen_kappa_score(rate2, rate3)
    return kappa12, kappa13, kappa23

waste_alighed_kappas = compute_kappa_score(kg_eval_df["R1_waste_aligned"].tolist(), kg_eval_df["R2_waste_aligned"].tolist(), kg_eval_df["R3_waste_aligned"].tolist())
waste_valid_kappas = compute_kappa_score(kg_eval_df["R1_waste_valid"].tolist(), kg_eval_df["R2_waste_valid"].tolist(), kg_eval_df["R3_waste_valid"].tolist())
resource_aligned_kappas = compute_kappa_score(kg_eval_df["R1_resource_aligned"].tolist(), kg_eval_df["R2_resource_aligned"].tolist(), kg_eval_df["R3_resource_aligned"].tolist())
resource_valid_kappas = compute_kappa_score(kg_eval_df["R1_resource_valid"].tolist(), kg_eval_df["R2_resource_valid"].tolist(), kg_eval_df["R3_resource_valid"].tolist())

avg_waste_aligned_kappa = sum(waste_alighed_kappas) / len(waste_alighed_kappas)
avg_waste_valid_kappa = sum(waste_valid_kappas) / len(waste_valid_kappas)
avg_resource_aligned_kappa = sum(resource_aligned_kappas) / len(resource_aligned_kappas)
avg_resource_valid_kappa = sum(resource_valid_kappas) / len(resource_valid_kappas)

print("Waste aligned kappas: ", waste_alighed_kappas)
print("Waste aligned kappa averaged: ", avg_waste_aligned_kappa)
print("Waste valid kappas: ", waste_valid_kappas)
print("Waste valid kappa averaged: ", avg_waste_valid_kappa)
print("Resource aligned kappas: ", resource_aligned_kappas)
print("Resource aligned kappa averaged: ", avg_resource_aligned_kappa)
print("Resource valid kappas: ", resource_valid_kappas)
print("Resource valid kappa averaged: ", avg_resource_valid_kappa)

# compute Fleiss' kappa
waste_aligned_fleiss_kappa = fleiss_kappa_three(kg_eval_df["R1_waste_aligned"].tolist(), kg_eval_df["R2_waste_aligned"].tolist(), kg_eval_df["R3_waste_aligned"].tolist())
waste_valid_fleiss_kappa = fleiss_kappa_three(kg_eval_df["R1_waste_valid"].tolist(), kg_eval_df["R2_waste_valid"].tolist(), kg_eval_df["R3_waste_valid"].tolist())
resource_aligned_fleiss_kappa = fleiss_kappa_three(kg_eval_df["R1_resource_aligned"].tolist(), kg_eval_df["R2_resource_aligned"].tolist(), kg_eval_df["R3_resource_aligned"].tolist())
resource_valid_fleiss_kappa = fleiss_kappa_three(kg_eval_df["R1_resource_valid"].tolist(), kg_eval_df["R2_resource_valid"].tolist(), kg_eval_df["R3_resource_valid"].tolist())
print("Waste aligned Fleiss' kappa: ", waste_aligned_fleiss_kappa)
print("Waste valid Fleiss' kappa: ", waste_valid_fleiss_kappa)
print("Resource aligned Fleiss' kappa: ", resource_aligned_fleiss_kappa)
print("Resource valid Fleiss' kappa: ", resource_valid_fleiss_kappa)


# ---------- majority voting ------------

def compute_voting_result(rate1, rate2, rate3):
    result_dict = {"0": 0, "1": 0, "2": 0, "3": 0}
    for i in range(len(rate1)):
        total_score = rate1[i] + rate2[i] + rate3[i]
        result_dict[str(total_score)] += 1

    return result_dict

def plot_voting_pie_chart(voting_dict, file_name):
    labels = list(voting_dict.keys())
    values = list(voting_dict.values())
    colors = ['#e6e6e6', '#bfbfbf', '#8c8c8c', '#737373']  # Light grey to dark grey
    plt.figure(figsize=(5, 5))
    plt.pie(values, colors=colors, autopct=lambda pct: f'{pct:.1f}%', textprops={'fontsize': 12}, startangle=140)
    plt.axis('equal')
    # plt.legend(labels, title=None, bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=4)
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
        
waste_aligned_voting_result = compute_voting_result(kg_eval_df["R1_waste_aligned"].tolist(), kg_eval_df["R2_waste_aligned"].tolist(), kg_eval_df["R3_waste_aligned"].tolist())
waste_valid_voting_result = compute_voting_result(kg_eval_df["R1_waste_valid"].tolist(), kg_eval_df["R2_waste_valid"].tolist(), kg_eval_df["R3_waste_valid"].tolist())
resource_aligned_voting_result = compute_voting_result(kg_eval_df["R1_resource_aligned"].tolist(), kg_eval_df["R2_resource_aligned"].tolist(), kg_eval_df["R3_resource_aligned"].tolist())
resource_valid_voting_result = compute_voting_result(kg_eval_df["R1_resource_valid"].tolist(), kg_eval_df["R2_resource_valid"].tolist(), kg_eval_df["R3_resource_valid"].tolist())
print(f"Waste aligned voting result: {waste_aligned_voting_result}, vote>=2: {waste_aligned_voting_result['2'] + waste_aligned_voting_result['3']}")
print(f"Waste valid voting result: {waste_valid_voting_result}, vote>=2: {waste_valid_voting_result['2'] + waste_valid_voting_result['3']}")
print(f"Resource aligned voting result: {resource_aligned_voting_result}, vote>=2: {resource_aligned_voting_result['2'] + resource_aligned_voting_result['3']}")
print(f"Resource valid voting result: {resource_valid_voting_result}, vote>=2: {resource_valid_voting_result['2'] + resource_valid_voting_result['3']}")

plot_voting_pie_chart(waste_aligned_voting_result, 'W2RKG_evaluation/kg_test_result_waste_aligned_voting.png')
plot_voting_pie_chart(waste_valid_voting_result, 'W2RKG_evaluation/kg_test_result_waste_valid_voting.png')
plot_voting_pie_chart(resource_aligned_voting_result, 'W2RKG_evaluation/kg_test_result_resource_aligned_voting.png')
plot_voting_pie_chart(resource_valid_voting_result, 'W2RKG_evaluation/kg_test_result_resource_valid_voting.png')