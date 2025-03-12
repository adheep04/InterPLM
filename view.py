import numpy as np
import script_args

# Replace 'path/to/your/file.npy' with the actual path to your .npy file
file_path = 'results/group1_features.npy'

# Load the NumPy array from the .npy file
loaded_array = np.load(file_path, allow_pickle=True)

# Print the loaded array
print(loaded_array)

# Check the shape of the array
print("Shape:", loaded_array.shape)

# Check the data type of the array
print("Data Type:", loaded_array.dtype)
