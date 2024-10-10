
# Predicción de Compras con Deep Learning

Este proyecto utiliza técnicas de Deep Learning para predecir si un cliente realizará una compra basada en atributos como edad, ingresos mensuales y puntaje de crédito.


## Estructura del Proyecto


prediccion_compra/
│
├── data/
│ └── clientes.csv # Datos de clientes.
│
├── model/
│ ├── model.py # Código del modelo.
│ └── utils.py # Funciones auxiliares.
│
├── requirements.txt # Dependencias del proyecto.
└── README.md # Documentación del proyecto.

## Evaluación del Modelo

Después de entrenar el modelo, se evaluó utilizando varias métricas:

- **Matriz de Confusión**: Proporciona una visión clara sobre cómo se clasificaron las predicciones en comparación con las etiquetas reales.
- **Informe de Clasificación**: Incluye precisión, recall y F1-score para cada clase.
- **Curva ROC**: Muestra la relación entre la tasa de verdaderos positivos y la tasa de falsos positivos.
- **Gráficos de Pérdida y Precisión**: Visualizan cómo cambiaron la pérdida y la precisión durante el entrenamiento.

Estas métricas permiten entender mejor el rendimiento del modelo y su capacidad para generalizar a datos no vistos.

## Instalación

1. Clona este repositorio.
2. Navega a la carpeta del proyecto.
3. Crea un entorno virtual (opcional pero recomendado).
4. Instala las dependencias:

```bash
pip install -r requirements.txt

Uso
Asegúrate de que el archivo clientes.csv esté en la carpeta data/.
Ejecuta el script principal:
bash
python model/model.py

Esto cargará los datos, entrenará el modelo y evaluará su precisión en un conjunto de prueba.
Contribuciones
Las contribuciones son bienvenidas. Siéntete libre de abrir issues o pull requests.
text

## Conclusión

Con esta estructura de proyecto bien organizada y siguiendo buenas prácticas de programación, podrás implementar un modelo de Deep Learning que prediga si un cliente realizará una compra o no basándose en sus atributos demográficos y financieros. ¡Buena suerte con tu implementación!