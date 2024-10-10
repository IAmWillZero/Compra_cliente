import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

def create_model(input_shape):
    """Crea y compila un modelo de red neuronal."""
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=input_shape),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, X_train, y_train, epochs=50):
    """Entrena el modelo con los datos proporcionados."""
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
    return history

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y muestra la matriz de confusión y el informe de clasificación."""
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Compra', 'Compra'], yticklabels=['No Compra', 'Compra'])
    plt.ylabel('Actual')
    plt.xlabel('Predicción')
    plt.title('Matriz de Confusión')
    plt.show()
    
    # Informe de clasificación
    report = classification_report(y_test, y_pred)
    print(report)

def plot_roc_curve(model, X_test, y_test):
    """Dibuja la curva ROC y calcula el AUC."""
    y_prob = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

def plot_training_history(histories):
    """Dibuja los gráficos de pérdida y precisión durante el entrenamiento para múltiples modelos."""
    
    for i, history in enumerate(histories):
        plt.figure(figsize=(12, 4))
        
        # Pérdida
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de Validación')
        plt.title(f'Pérdida del Modelo {i+1}')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        
        # Precisión
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
        plt.title(f'Precisión del Modelo {i+1}')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        
        plt.show()

def cross_validate_model(X, y):
   """Realiza validación cruzada para ajustar hiperparámetros."""
   kf = KFold(n_splits=5)  # Dividir en 5 pliegues
   accuracies = []
   histories = []
   
   for fold_index, (train_index, val_index) in enumerate(kf.split(X)):
       print(f"\nEntrenando en pliegue {fold_index + 1}...")
       X_train_fold, X_val_fold = X[train_index], X[val_index]
       y_train_fold, y_val_fold = y[train_index], y[val_index]
       
       model = create_model((X_train_fold.shape[1],))
       history = train_model(model, X_train_fold, y_train_fold)
       histories.append(history)

       test_loss, test_acc = model.evaluate(X_val_fold, y_val_fold)
       accuracies.append(test_acc)
       print(f'Precisión en pliegue {fold_index + 1}: {test_acc:.4f}')
       
   print(f'\nPrecisión promedio: {np.mean(accuracies):.4f}')
   return histories

def ensemble_models(models):
   """Combina varios modelos para mejorar la precisión."""
   def predict_ensemble(X):
       predictions = np.array([model.predict(X) for model in models])
       return np.round(np.mean(predictions, axis=0)).astype(int)

   return predict_ensemble

def predict_new_data(model, new_data):
    """Realiza predicciones sobre nuevos datos."""
    predictions = (model.predict(new_data) > 0.5).astype("int32")
    return predictions

def main():
   # Cargar los datos desde el archivo CSV.
   df = pd.read_csv('data/clientes.csv')

   # Separar las características (X) y la variable objetivo (y).
   X = df[['edad', 'ingresos_mensuales', 'puntaje_credito']].values
   y = df['compra'].values

   # Realizar validación cruzada para ajustar hiperparámetros.
   histories = cross_validate_model(X,y)

   # Dividir los datos en conjuntos de entrenamiento y prueba.
   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

   # Crear múltiples modelos para el ensemble.
   models = [create_model((X_train.shape[1],)) for _ in range(5)]
   
   for model in models:
       history = train_model(model,X_train,y_train)

   # Evaluar el ensemble.
   ensemble_predictor = ensemble_models(models)
   predictions = ensemble_predictor(X_test)

   # Calcular métricas del ensemble.
   evaluate_model(models[0], X_test,y_test)  # Solo se usa uno para mostrar la matriz.

   # Nuevos datos para hacer predicciones (asegúrate de que tengan el mismo formato)
   new_data = np.array([
       [30, 3500, 700],  # Ejemplo 1
       [45, 6000, 750],  # Ejemplo 2
       [22, 1500, 580],  # Ejemplo 3
       [50, 4000, 680]   # Ejemplo 4
   ])

   # Realizar predicciones sobre los nuevos datos
   new_predictions = predict_new_data(models[0], new_data)

   # Mostrar resultados
   for i in range(len(new_data)):
       print(f"Ejemplo {i+1}: {'Compra' if new_predictions[i][0] == 1 else 'No Compra'}")

if __name__ == "__main__":
   main()