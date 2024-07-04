#necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

#setting seed ensures reproducibility
np.random.seed(3)

# Load and preprocess the dataset
data = pd.read_csv('Iris-flower-Classification/IRIS.csv')
X = data.drop(['species'], axis=1)
y = data['species']

# Encode labels and one-hot encode targets
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code for Neural Network Model
class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_outputs, learning_rate):
    
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs 
        self.learning_rate = learning_rate

        #random value declaration format: {output, input}
        #initial randomized values before training for weights and biases
        self.weights = []
        self.weights.append(np.random.randn(self.n_hidden, self.n_input))
        self.weights.append(np.random.randn(self.n_outputs, self.n_hidden))
        
        self.biases = []
        self.biases.append(np.random.randn(self.n_hidden))
        self.biases.append(np.random.randn(self.n_outputs))

    # activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of activation function
    def dsigmoid(self, x):
        return x * (1 - x)

    # dot product to calculate output
    def make_prediction(self, input_vector):
        hidden_layer = self.sigmoid(np.dot(self.weights[0], input_vector) + self.biases[0])
        output_layer = self.sigmoid(np.dot(self.weights[1], hidden_layer) + self.biases[1])
        return output_layer

    def gradient_descent(self, input_vector, target):
        # Forward pass
        hidden_layer = self.sigmoid(np.dot(self.weights[0], input_vector) + self.biases[0])
        output_layer = self.sigmoid(np.dot(self.weights[1], hidden_layer) + self.biases[1])
        
        # Calculate error
        error = output_layer - target
        
        # Derivatives of error used to find gradients using chain rule
        derror_doutput = error * self.dsigmoid(output_layer)
        doutput_dhidden = np.dot(self.weights[1].T, derror_doutput) * self.dsigmoid(hidden_layer)

        # Calculate gradients
        derror_dweights1 = np.outer(derror_doutput, hidden_layer)
        derror_dbias1 = derror_doutput
        derror_dweights0 = np.outer(doutput_dhidden, input_vector)
        derror_dbias0 = doutput_dhidden

        return [derror_dweights0, derror_dweights1], [derror_dbias0, derror_dbias1]

    #second half of backpropagation, updating weights and biases according to gradients
    def update(self, gradients, biases):
        self.weights[0] -= self.learning_rate * gradients[0]
        self.weights[1] -= self.learning_rate * gradients[1]
        self.biases[0] -= self.learning_rate * biases[0]
        self.biases[1] -= self.learning_rate * biases[1]

    def train(self, input_vectors, targets, iterations):
        all_errors = []
        for current_iteration in range(iterations):
            # Pick a random data point
            random_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_index]
            target = targets[random_index]

            # Compute gradients using backpropagation
            gradients, biases = self.gradient_descent(input_vector, target)

            # Update weights and biases
            self.update(gradients, biases)

            # Calculate cumulative error for monitoring
            if current_iteration % 100 == 0:
                cumulative_error = 0
                for i in range(len(input_vectors)):
                    data_point = input_vectors[i]
                    target = targets[i]
                    error = np.square(self.make_prediction(data_point) - target)
                    cumulative_error += error
                all_errors.append(np.sum(cumulative_error))

        return all_errors

# Initialize neural network
n_input = 4
n_hidden = 2
n_outputs = 3
learning_rate = 0.1
neural_network = NeuralNetwork(n_input=n_input, n_hidden=n_hidden, n_outputs=n_outputs, learning_rate=learning_rate)

# Train neural network
epoch = 10000
errors = neural_network.train(X_train, y_train, epoch)

# Plot error over iterations
plt.plot(errors)
plt.xlabel("Iterations (in 100s)")
plt.ylabel("Cumulative Error")
plt.title("Error over Iterations")
plt.show()

# Evaluate neural network on test set
y_pred = []
for i in range(len(X_test)):
    prediction = neural_network.make_prediction(X_test[i])
    y_pred.append(np.argmax(prediction))

y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Confusion matrix
cf_matrix = confusion_matrix(y_test_labels, y_pred)

# Labels for the confusion matrix heatmap
labels = ['Correct Classification', 'Misclassification', 'Misclassification', 
          'Misclassification', 'Correct Classification', 'Misclassification', 
          'Misclassification', 'Misclassification', 'Correct Classification']
values = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
annotations = [f'{v1}\n{v2}' for v1, v2 in zip(labels, values)]
annotations = np.asarray(annotations).reshape(3, 3)

# Axis labels for heatmap
axxlabels = ['Predicted Setosa', 'Predicted Versicolor', 'Predicted Virginica']
axylabels = ['Actual Setosa', 'Actual Versicolor', 'Actual Virginica']

# Plotting the heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cf_matrix, annot=annotations, fmt='', xticklabels=axxlabels, yticklabels=axylabels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification report
target_names = ['setosa', 'versicolor', 'virginica']
print(classification_report(y_test_labels, y_pred, target_names=target_names))