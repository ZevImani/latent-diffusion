import os
import numpy as np

# Define the directory containing the .npy files
directory = 'protons64_ae_disc_post_test/'

# Initialize an empty list to store the arrays
combined_data = []


# Iterate through all files in the directory
for filename in sorted(os.listdir(directory)):
	if filename.endswith('.npy') and 'posterior_batch_' in filename:
		# Load the .npy file
		file_path = os.path.join(directory, filename)
		data = np.load(file_path)
		
		# Append the data to the list
		combined_data.append(data)
		

# Combine all arrays into one
combined_data = np.concatenate(combined_data, axis=0)

# Save the combined array into a new .npy file
np.save(directory[:-1]+".npy", combined_data)

print("Saved:", directory[:-1]+".npy, shape =", combined_data.shape)
