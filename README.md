# Proyecto-Machine-Learning

# Análisis de sentimientos en reseñas de Yelp con transfer learning

Proyecto final de Machine Learning.
**Autora:** Stephanía Cárdenas

## Descripción

Clasificación binaria del sentimiento (negativo / positivo) de reseñas de Yelp.
Las reseñas se representan con embeddings de un modelo de lenguaje preentrenado
(transfer learning por extracción de características) y se comparan dos
clasificadores: regresión logística y SVM con kernel RBF.

## Dataset

**Yelp Review Full** (Zhang, Zhao y LeCun, 2015), disponible en Hugging Face:
https://huggingface.co/datasets/Yelp/yelp_review_full

El dataset no se incluye en este repositorio; la notebook lo descarga
automáticamente.

- Etiqueta binaria: 1–2 estrellas = negativo, 4–5 estrellas = positivo
  (se excluyen las reseñas de 3 estrellas).
- Submuestra balanceada: 10.000 reseñas de entrenamiento y 2.000 de prueba
  (partición de prueba oficial).

## Metodología

1. Análisis exploratorio de datos (EDA).
2. Extracción de características con el modelo preentrenado `all-MiniLM-L6-v2`
   (embeddings de 384 dimensiones).
3. Pipelines de scikit-learn: escalado estándar + clasificador.
4. Optimización de hiperparámetros con validación cruzada repetida (2 × 5).
5. Comparación estadística (regresión logística vs. SVM) con la prueba de Wilcoxon.
6. Evaluación final en el conjunto de prueba y análisis de errores.
7. Visualización del espacio de características con PCA y t-SNE.

## Resultados principales

| Modelo | Accuracy (validación cruzada) |
|---|---|
| Línea base (clase más frecuente) | 0,500 |
| Regresión logística (C = 0,01) | 0,877 ± 0,009 |
| SVM RBF (C = 1) | 0,880 ± 0,010 |

- Prueba de Wilcoxon: p = 0,1934, sin diferencia significativa entre modelos.
- Modelo final (regresión logística) en el conjunto de prueba:
  **accuracy = 0,879**.

## Cómo ejecutar

1. Abrir `Proyecto_ML_Analisis_de_Sentimientos_TRANSFER_LEARNING.ipynb` en Google Colab.
2. Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
3. Ejecutar todas las celdas: *Entorno de ejecución → Ejecutar todas*.

Librerías principales: pandas, numpy, matplotlib, scikit-learn, scipy,
sentence-transformers y wordcloud.
