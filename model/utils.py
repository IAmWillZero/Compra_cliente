import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Carga los datos desde un archivo CSV."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """Preprocesa los datos: separa características y etiquetas, y normaliza."""
    X = data[['edad', 'ingresos_mensuales', 'puntaje_credito']]
    y = data['compra']
    
    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)