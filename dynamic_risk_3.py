import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Define the linguistic terms and their associated fuzzy ranges for the component attributes
component_attributes = {
    'Confidentiality': ['Low', 'Medium', 'High'],
    'Integrity': ['Low', 'Medium', 'High'],
    'Availability': ['Low', 'Medium', 'High']
}

term_ranges = {
    'Low': [0, 0, 5],
    'Medium': [0, 5, 10],
    'High': [5, 10, 10]
}

# Define the device attributes and their possible values
device_attributes = {
    'Confidentiality': ['Low', 'Medium', 'High'],
    'Integrity': ['Low', 'Medium', 'High'],
    'Availability': ['Low', 'Medium', 'High']
}

# Define the weights for the component attributes
weights = np.array([0.5, 0.3, 0.2])

# Define the component fuzzy sets for the batch of SCADA components
component_fuzzy_sets = {}
for attr in component_attributes:
    component_fuzzy_sets[attr] = {}
    for term in component_attributes[attr]:
        component_fuzzy_sets[attr][term] = fuzz.trimf(np.arange(0, 11, 1), term_ranges[term])

# Generate a batch of SCADA component data
batch_data = np.random.randint(0, 11, (3, 5), dtype=np.int)

# Initialize the aggregated weighted score for the batch of SCADA components
aggregated_weighted_score = np.zeros(len(device_attributes))

# Initialize the aggregated weighted matrix for the batch of SCADA components
aggregated_weighted_matrix = np.zeros(len(component_attributes))

# Calculate the normalized matrix for the batch of SCADA components
normalized_matrix = np.zeros((len(component_attributes), batch_data.shape[1]))
for i in range(batch_data.shape[1]):
    confidentiality_index = batch_data[0, i].clip(max=len(component_attributes['Confidentiality'])-1)
    normalized_matrix[0, i] = np.sum(component_fuzzy_sets['Confidentiality'][component_attributes['Confidentiality'][confidentiality_index]][batch_data[:, i]])

    integrity_index = batch_data[1, i].clip(max=len(component_attributes['Integrity'])-1)
    normalized_matrix[1, i] = np.sum(component_fuzzy_sets['Integrity'][component_attributes['Integrity'][integrity_index]][batch_data[:, i]])
    availability_index = batch_data[2, i].clip(max=len(component_attributes['Availability'])-1)
    normalized_matrix[2, i] = np.sum(component_fuzzy_sets['Availability'][component_attributes['Availability'][availability_index]][batch_data[:, i]])

# Choose the security attributes for the device
chosen_device_attributes = {
'Confidentiality': 'High',
'Integrity': 'Medium',
'Availability': 'Low'
}

# Calculate the fuzzy scores for the device
device_fuzzy_scores = {}
for attr in device_attributes:
  device_fuzzy_scores[attr] = {}
  for term in device_attributes[attr]:
    idx = np.where(np.array(component_attributes[attr]) == term)[0]
    if len(idx) > 0:
      device_fuzzy_scores[attr][term] = np.max(normalized_matrix[idx, :], initial=0)
    else:
      device_fuzzy_scores[attr][term] = 0

# Calculate the weighted score for the device
weighted_device_score = np.zeros_like(weights, dtype=float)
for i, attr in enumerate(device_attributes):
  term = chosen_device_attributes[attr]
  idx = np.where(np.array(component_attributes[attr]) == term)[0]
  if len(idx) > 0:
    weighted_device_score += device_fuzzy_scores[attr][term] * weights[i]

#Calculate the aggregated weighted score for the batch of SCADA components
for i, attr in enumerate(device_attributes):
  term = chosen_device_attributes[attr]
  idx = np.where(np.array(component_attributes[attr]) == term)[0]
  if len(idx) > 0:
    aggregated_weighted_score[i] = np.max(weighted_matrix[idx, :], initial=0)

#Calculate the weighted matrix for the batch of SCADA components
weighted_matrix = normalized_matrix * weights.reshape(-1, 1)

#Calculate the aggregated weighted matrix for the batch of SCADA components
aggregated_weighted_matrix = np.max(weighted_matrix, axis=0)

#Visualize the fuzzy scores for the batch of SCADA components
fig, axs = plt.subplots(len(component_attributes), batch_data.shape[1], figsize=(10, 10), sharex=True, sharey=True)
for j in range(batch_data.shape[1]):
  for k, attr in enumerate(component_attributes):
    for l, term in enumerate(component_attributes[attr]):
      axs[k, j].plot(component_fuzzy_sets[attr][term], label=term)
      axs[k, j].fill_between(np.arange(0,11, 1), component_fuzzy_sets[attr][term], alpha=0.1)
      axs[k, j].legend()
      axs[k, j].set_ylim([0, 1])
      axs[k, j].set_xlabel('Component {}'.format(j + 1))
      axs[k, j].set_ylabel('Membership')
plt.tight_layout()
plt.show()

# Visualize the device fuzzy scores and weighted score
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i, attr in enumerate(device_attributes):
  axs[0].barh(np.arange(len(device_attributes[attr])), [device_fuzzy_scores[attr][term] for term in device_attributes[attr]], height=0.5, align='center', label=attr)
  axs[0].set_yticks(np.arange(len(device_attributes[attr])))
  axs[0].set_yticklabels(device_attributes[attr])
  axs[0].set_xlabel('Fuzzy Score')
  axs[0].set_title('Device Fuzzy Scores')
  axs[0].legend()
  axs[1].barh([0], weighted_device_score, height=0.5, align='center')
  axs[1].set_yticks([0])
  axs[1].set_yticklabels(['Weighted Score'])
  axs[1].set_xlabel('Weighted Score')
  axs[1].set_title('Device Weighted Score')
plt.tight_layout()
plt.show()

# Visualize the aggregated weighted matrix and score for the batch of SCADA components
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(weighted_matrix, cmap='gray_r')
axs[0].set_xticks(np.arange(batch_data.shape[1]))
axs[0].set_yticks(np.arange(len(component_attributes)))
axs[0].set_xticklabels(['Component {}'.format(i + 1) for i in range(batch_data.shape[1])])
axs[0].set_yticklabels(component_attributes.keys())
axs[0].set_xlabel('Component')
axs[0].set_ylabel('Attribute')
for i in range(len(component_attributes)):
  for j in range(batch_data.shape[1]):
    axs[0].text(j, i, '{:.2f}'.format(weighted_matrix[i, j]), ha='center', va='center', color='w')
    axs[0].set_title('Weighted Matrix')
    axs[1].barh(np.arange(len(component_attributes)), aggregated_weighted_matrix.reshape(-1,1), align='center')
    axs[1].set_linewidth(1.0)
    axs[1].set_yticks(np.arange(len(component_attributes)))
    axs[1].set_yticklabels(component_attributes.keys())
    axs[1].set_xlabel('Aggregated Weighted Score')
    axs[1].set_title('Aggregated Weighted Scores')
plt.tight_layout()
plt.show()

# Visualize the aggregated weighted score for the batch
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.barh(np.arange(len(device_attributes)), aggregated_weighted_score, height=np.ones(len(component_attributes)), align='center')
ax.set_yticks(np.arange(len(device_attributes)))
ax.set_yticklabels(device_attributes.keys())
ax.set_xlabel('Weighted Score')
ax.set_title('Aggregated Weighted Score')
plt.tight_layout()
plt.show()