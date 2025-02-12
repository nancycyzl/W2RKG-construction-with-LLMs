import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


kg_eval_path = "result_all/after_fusion_v2/thre08_complete/kg_test_dataset_combined.xlsx"
kg_eval_df = pd.read_excel(kg_eval_path)

print(kg_eval_df.shape)

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
plt.savefig('result_all/after_fusion_v2/thre08_complete/kg_test_result_waste.png', 
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
plt.savefig('result_all/after_fusion_v2/thre08_complete/kg_test_result_resource.png', 
            bbox_inches='tight', dpi=300)
plt.close('all')
