import pandas as pd
import numpy as np

# Configuración inicial.
np.random.seed(42) # Para reproducibilidad.

# Número total de muestras.
num_samples = 500 

# Generar edades entre (18-70).
edad = np.random.randint(18, high=70,size=num_samples)

# Generar ingresos mensuales basados en la edad.
ingresos_mensuales = np.clip(np.random.normal(loc=3000 + (edad -30) * (500), scale=1000), a_min=1000,a_max=10000)

# Generar puntaje de crédito entre (300-850).
puntaje_credito = np.clip(np.random.normal(loc=600 + (edad -30) * (5), scale=50), a_min=300,a_max=850)

# Generar la variable objetivo 'compra' basada en ingresos y puntajes de crédito.
compra = np.where((ingresos_mensuales > 4000) & (puntaje_credito > 600), np.random.choice([1], size=num_samples),
                  np.random.choice([0], size=num_samples))

# Crear DataFrame.
data = {
    "edad": edad,
    "ingresos_mensuales": ingresos_mensuales,
    "puntaje_credito": puntaje_credito,
    "compra": compra,
}

df = pd.DataFrame(data)

# Guardar como CSV.
df.to_csv('data/clientes.csv', index=False)

print("Archivo 'clientes.csv' creado con éxito.")