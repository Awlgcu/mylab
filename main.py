import numpy as np

def hebbian_learning(input_pattern, weight_matrix, learning_rate):
    """
    Applies Hebbian learning to update the weight matrix based on the input pattern.

    Parameters:
    - input_pattern (numpy array): The input pattern as a 1D numpy array.
    - weight_matrix (numpy array): The current weight matrix.
    - learning_rate (float): The learning rate for Hebbian learning.

    Returns:
    - updated_weight_matrix (numpy array): The updated weight matrix.
    """

    # Convert input pattern to a column vector
    input_pattern = np.reshape(input_pattern, (len(input_pattern), 1))

    # Update the weight matrix using Hebbian learning rule
    updated_weight_matrix = weight_matrix + learning_rate * np.dot(input_pattern, input_pattern.T)

    return updated_weight_matrix

# Example usage:
input_pattern1 = np.array([1, -1, 1])
input_pattern2 = np.array([-1, 1, -1])

initial_weight_matrix = np.zeros((len(input_pattern1), len(input_pattern1)))

learning_rate = 0.1

# Apply Hebbian learning for the first input pattern
updated_weight_matrix = hebbian_learning(input_pattern1, initial_weight_matrix, learning_rate)

# Apply Hebbian learning for the second input pattern
updated_weight_matrix = hebbian_learning(input_pattern2, updated_weight_matrix, learning_rate)

print("Initial Weight Matrix:")
print(initial_weight_matrix)
print("\nUpdated Weight Matrix after learning:")
print(updated_weight_matrix)