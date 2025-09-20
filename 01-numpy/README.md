# ml-learning-journey
# Intro to NumPy — Practice Notebook
This notebook is my personal practice for learning NumPy, following the ML Labs Intro to NumPy exercises.  
It covers array reshaping, sigmoid, softmax, dot/outer products, losses, and matrix operations.  
What I learnt:  

•how to reshape a 3D array into a vector  
def convert_to_vector(array):  
    vector = array.reshape((array.shape[0] * array.shape[1] * array.shape[2], 1))  
    return vector  
    
• sigmoid function - used to map data into an S-shaped curve  
def sigmoid_np_exp(x):  
    return 1 / (1 + np.exp(-x))  
    
• how math.exp() and np.exp() are different(np.exp() can take a numpy array, math.exp() can only take 1 input)  

• softmax function - converts a tuple of K real numbers into a probability distribution over K possible outcomes  
def sigmoid_np_exp(x):  
    return 1 / (1 + np.exp(-x))  