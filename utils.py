import os

def load_data(filepath):
    """
    Carga un archivo CSV plano donde cada línea es un vector numérico separado por comas.
    Retorna una lista de listas de floats o None si ocurre un error.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] El archivo '{filepath}' no existe. Verifica la ruta.")
        return None

    data = []
    try:
        with open(filepath, 'r') as file:
            for linea in file:
                if linea.strip():  # Evita procesar líneas vacías
                    elementos = linea.strip().split(',')
                    try:
                        vector = [float(x) for x in elementos]
                        data.append(vector)
                    except ValueError:
                        print(f"[ERROR] Línea mal formateada: {linea.strip()}")
                        return None

        if len(data) == 0:
            print(f"[ERROR] El archivo '{filepath}' está vacío o mal formateado.")
            return None

        return data

    except Exception as e:
        print(f"[ERROR] Ocurrió un problema al leer '{filepath}': {e}")
        return None