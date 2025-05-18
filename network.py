import numpy as np
from graph import plot_accuracy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)  # Asumiendo que x ya es sigmoid(x)

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer):
        np.random.seed(42)
        self.layers = []
        self.biases = []
        self.activations = []

        # Capa de entrada → primera capa oculta
        self.layers.append(np.random.uniform(-1, 1, (neurons_per_layer, input_size)))
        self.biases.append(np.random.uniform(-1, 1, (neurons_per_layer, 1)))

        # Capas ocultas intermedias
        for _ in range(hidden_layers - 1):
            self.layers.append(np.random.uniform(-1, 1, (neurons_per_layer, neurons_per_layer)))
            self.biases.append(np.random.uniform(-1, 1, (neurons_per_layer, 1)))

        # Última capa oculta → capa de salida
        self.layers.append(np.random.uniform(-1, 1, (output_size, neurons_per_layer)))
        self.biases.append(np.random.uniform(-1, 1, (output_size, 1)))

    def forward(self, x):
        a = np.array(x).reshape(-1, 1)
        self.activations = [a]

        for w, b in zip(self.layers, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            self.activations.append(a)

        return self.activations[-1].flatten()

    def backward(self, x, y, learning_rate=0.001):
        # Propagación hacia adelante
        self.forward(x)
        y = np.array(y).reshape(-1, 1)

        # Inicializar lista de deltas
        deltas = [None] * len(self.layers)

        # Capa de salida
        error = self.activations[-1] - y
        deltas[-1] = error * sigmoid_deriv(self.activations[-1])

        # Capas ocultas hacia atrás
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = np.dot(self.layers[i + 1].T, deltas[i + 1]) * sigmoid_deriv(self.activations[i + 1])

        # Actualización de pesos y sesgos
        for i in range(len(self.layers)):
            dw = np.dot(deltas[i], self.activations[i].T)
            self.layers[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * deltas[i]

    def train(self, X, y, X_test, y_test, epochs):
        train_acc_list = []
        test_acc_list = []

        for epoca in range(epochs):
            for xi, yi in zip(X, y):
                self.backward(xi, yi)

            # Evaluar precisión
            train_acc = self.evaluate(X, y)
            test_acc = self.evaluate(X_test, y_test)

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print(f"Época {epoca + 1}: Precisión entrenamiento = {train_acc:.2f}%, Precisión prueba = {test_acc:.2f}%")

        # Mostrar gráfico al final del entrenamiento
        plot_accuracy(train_acc_list, test_acc_list)

    def predict(self, X):
        return [self.forward(x) for x in X]

    def evaluate(self, X, y):
        predicciones = self.predict(X)
        correctos = 0
        for pred, esperado in zip(predicciones, y):
            pred_bin = [1 if p >= 0.5 else 0 for p in pred]
            esp_bin = [int(e) for e in esperado]
            if pred_bin == esp_bin:
                correctos += 1
        return (correctos / len(y)) * 100

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)