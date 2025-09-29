# ml-learning-journey
# Intro to NumPy — Practice Notebook
This notebook is my personal practice for learning NumPy, following the ML Labs Intro to NumPy exercises.  
It covers array reshaping, sigmoid, softmax, dot/outer products, losses, and matrix operations.  
What I learnt:  

• How to reshape a 3D array into a vector  
def convert_to_vector(array):  
    vector = array.reshape((array.shape[0] * array.shape[1] * array.shape[2], 1))  
    return vector  
    
• Sigmoid Function - used to map data into an S-shaped curve  
def sigmoid_np_exp(x):  
    return 1 / (1 + np.exp(-x))  
    - used to map any number into a probability on a scale between 0 and 1 depending on its value  
    
• how math.exp() and np.exp() are different(np.exp() can take a numpy array, math.exp() can only take 1 input)  

• Softmax Function - converts a tuple of K real numbers into a probability distribution over K possible outcomes  
def sigmoid_np_exp(x):  
    return 1 / (1 + np.exp(-x)) 
    - interprets values as probabilities against each other so they all add up to 1  
  
• Dot product - the combination of vectors through multiplication in order to get a single overall vector  
- math - magnitude of vector A * magnitude of vector B * cos(theta)  
- or Ax * By + Bx * Ay  
- both vectors must be the same size  
def dot_product_numpy(vector1, vector2):  
    dot_product = np.dot(vector1, vector2)  
    return dot_product  
  
• Outer Product - creates a matrix by multiplying each element in vector A by all elements in vector B
- vectors do not need to be the same size  
def outer_product_numpy(vector1, vector2):
    outer_product = np.outer(vector1, vector2)
    return outer_product
  
• L1 and L2 loss - average of the absolute difference between the predicted and actual values  
- ie how far off you are from the true value on average ignoring if it's over or under  
- also known as MAE(Mean Absolute Error) and MSE(Mean Squared Error) respectively  
- vectors must be the same size  
- L2 is used for harsher punishments for training models as the value they are given that they are off by is exponentially higher  
  
• Matrix Addition - adds 2 matrices element-wise  
def matrix_addition_numpy(matrix1, matrix2):  
    if matrix1.shape != matrix2.shape:  
        raise ValueError("Matrices must have the same dimensions for addition")  
  
    result_matrix = matrix1 + matrix2  
    return result_matrix 
  
• Matrix Multiplication - not element-wise, more complex  
- A @ B or np.matmul(A,B)  
def matrix_multiplication_numpy(matrix1, matrix2):  
    if matrix1.shape[1] != matrix2.shape[0]:  
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix")  
  
    result_matrix = np.dot(matrix1, matrix2)  
    return result_matrix  