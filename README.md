# EP1 DLY0100 – MLP para predecir contratación de depósitos (Bank Marketing)

Red neuronal multicapa (MLP) en **PyTorch** que predice si un cliente bancario contratará un depósito a plazo (`y = yes/no`), usando el dataset *Bank Marketing* de UCI (45.211 clientes, 16 variables).

## Contenido

| Archivo | Descripción |
|---|---|
| `EP1_MLP_BankMarketing.ipynb` | **Notebook de entrega (informe)**: EDA, preprocesamiento, funciones programadas desde cero, experimentos, modelo final, métricas y conclusiones |
| `bank_marketing.csv` | Dataset entregado por el docente |
| `Evaluacion_MLP_Dataset_BankMarketing.ipynb` | Template original del docente |
| `requirements.txt` | Dependencias |

## Cómo ejecutar

**Google Colab (recomendado)**
1. Abrir `EP1_MLP_BankMarketing.ipynb` en Colab (*Archivo → Subir notebook* o desde GitHub).
2. Subir `bank_marketing.csv` al panel de archivos. Si no se sube, el notebook lo descarga automáticamente desde UCI.
3. *Entorno de ejecución → Ejecutar todas*. Todas las librerías vienen preinstaladas en Colab.

**Local**
```bash
pip install -r requirements.txt
jupyter notebook EP1_MLP_BankMarketing.ipynb
```
La ejecución completa toma ~10–15 minutos en CPU (incluye todos los experimentos y la validación cruzada). Los resultados son reproducibles (semilla fija `SEED = 42`).

## Estructura del notebook

0. Configuración del entorno · 1. Carga · 2. Diccionario · 3. EDA · 4. Preprocesamiento
5. Funciones de activación, error y salida desde cero (NumPy) + perceptrón
6. Definición del MLP en PyTorch
7. Experimentos controlados: épocas, learning rate, batch size
8. Comparación de activaciones · 9. Comparación de pérdidas
10. Regularización (dropout, batch norm, L2) y optimizadores
11. Arquitectura y búsqueda de hiperparámetros
12. Modelo final, umbral de decisión y evaluación en test
13. Validación cruzada estratificada (5 folds)
14. Análisis crítico y propuestas de mejora · 15. Conclusiones · 16. Referencias

## Decisiones principales

- Se **excluye `duration`** (sólo se conoce después de la llamada → fuga de información).
- Desbalance 88/12 → métricas F1 / Precision / Recall / PR-AUC, **BCE ponderada** y **umbral ajustado en validación**.
- Preprocesamiento ajustado **sólo con train** (StandardScaler + One-Hot) para evitar *data leakage*.
- ReLU en capas ocultas, sigmoide en la salida, Adam, Dropout + L2 + early stopping.
"# EP1_MLP_BankMarketing" 
