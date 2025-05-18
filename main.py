#Ernesto Carmona
#Sección 1: Computación Emergente
#Tarea 2: Perceptrón Multicapa

from network import NeuralNetwork
from utils import load_data
import pickle

def main():
    print("=== Perceptrón Multicapa ===")
    print("1. Crear un nuevo perceptrón")
    print("2. Cargar perceptrón desde archivo")
    opcion = input("Seleccione una opción: ")

    if opcion == '1':
        red = crear_nueva_red()
        entrenamiento_inicial(red)
    elif opcion == '2':
        ruta = input("Ingrese la RUTA del archivo de red guardada (*.pkl): ")
        with open(ruta, 'rb') as f:
            red = pickle.load(f)
        print("Red cargada exitosamente.")
    else:
        print("Opción inválida.")
        return

    menu_operaciones(red)

def crear_nueva_red():
    input_size = int(input("Número de neuronas de entrada: "))
    output_size = int(input("Número de neuronas de salida: "))
    hidden_layers = int(input("Número de capas ocultas: "))
    neurons_per_layer = int(input("Número de neuronas por capa: "))

    red = NeuralNetwork(input_size, output_size, hidden_layers, neurons_per_layer)
    return red

def entrenamiento_inicial(red):
    archivo_X = input("Ingrese la RUTA del archivo de entrenamiento (X): ")
    archivo_y = input("Ingrese la RUTA del archivo de salidas esperadas (y): ")
    archivo_X_test = input("Ingrese la RUTA del archivo de prueba (X): ")
    archivo_y_test = input("Ingrese la RUTA del archivo de salidas esperadas de prueba (y): ")
    epocas = int(input("Número de épocas: "))

    X = load_data(archivo_X)
    if X is None:
        print("No se pudo cargar el archivo de entrenamiento (X).")
        return

    y = load_data(archivo_y)
    if y is None:
        print("No se pudo cargar el archivo de salidas esperadas (y).")
        return

    X_test = load_data(archivo_X_test)
    if X_test is None:
        print("No se pudo cargar el archivo de prueba (X_test).")
        return

    y_test = load_data(archivo_y_test)
    if y_test is None:
        print("No se pudo cargar el archivo de salidas esperadas de prueba (y_test).")
        return

    # Si todo cargó correctamente, procede al entrenamiento
    red.train(X, y, X_test, y_test, epocas)

def menu_operaciones(red):
    print('Entrando al menú')
    while True:
        print("\n--- Menú de operaciones ---")
        print("1. Ejecutar la red")
        print("2. Seguir entrenando")
        print("3. Guardar red en archivo")
        print("4. Salir")
        opcion = input("Seleccione una opción: ")

        if opcion == '1':
            ejecutar_red(red)
        elif opcion == '2':
            entrenamiento_inicial(red)
        elif opcion == '3':
            nombre_archivo = input("Nombre para guardar la red (*.pkl): ")
            with open(nombre_archivo, 'wb') as f:
                pickle.dump(red, f)
            print("Red guardada exitosamente.")
        elif opcion == '4':
            break
        else:
            print("Opción inválida.")

def ejecutar_red(red):
    print("1. Introducir vector por teclado")
    print("2. Cargar archivo de prueba")
    opcion = input("Seleccione una opción: ")
    if opcion == '1':
        entrada = input("Ingrese los valores separados por comas: ")
        vector = [float(x) for x in entrada.strip().split(',')]
        salida = red.forward(vector)
        print("Salida de la red:", salida)
    elif opcion == '2':
        archivo = input("Ruta del archivo: ")
        X = load_data(archivo)
        for vector in X:
            salida = red.forward(vector)
            print("Entrada:", vector, "→ Salida:", salida)

if __name__ == "__main__":
    main()