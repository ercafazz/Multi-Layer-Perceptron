import matplotlib.pyplot as plt

def plot_accuracy(train_acc, test_acc):
    """
    Recibe dos listas con la precisión por época (entrenamiento y prueba)
    y muestra un gráfico de líneas comparando ambas curvas.
    """
    epocas = list(range(1, len(train_acc) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epocas, train_acc, label='Entrenamiento', marker='o')
    plt.plot(epocas, test_acc, label='Prueba', marker='s')
    plt.title('Precisión por Época')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()