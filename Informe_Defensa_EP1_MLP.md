# Informe de Defensa — EP1 MLP Bank Marketing (DLY0100)

> Guía para preparar la presentación/defensa oral (10 min). Explica **qué hace el notebook**, **por qué se hizo cada cosa** y **cómo responder preguntas del docente**. Basado en el notebook `EP1_MLP_BankMarketing.ipynb`.

---

## 0. El proyecto en 30 segundos (elevator pitch)

Construimos un **MLP (perceptrón multicapa) en PyTorch** que predice si un cliente de un banco portugués **contratará un depósito a plazo** cuando lo llaman por teléfono. Problema de **clasificación binaria** sobre 45.211 clientes y 16 variables.

Puntos de venta del trabajo:
- Preprocesamiento **sin fuga de información** y justificado desde un EDA.
- Funciones de activación, error y salida **programadas desde cero en NumPy** y validadas contra PyTorch.
- **Experimentos controlados** (un parámetro a la vez) con evidencia visual.
- Manejo correcto del **desbalance** (BCE ponderada + umbral óptimo).
- Métricas interpretadas con **criterio de negocio**, no solo accuracy.
- Validación cruzada + test separado e intacto.

---

## 1. El problema y los datos

- **Banco portugués**, campaña de marketing telefónico (dataset UCI *Bank Marketing*, Moro et al., 2014).
- **Objetivo**: decidir **a quién llamar** previendo si contratará el depósito (clase `y = yes`, ~11,7 % de los casos).
- Variables: 7 numéricas (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`), 9 categóricas.
- `duration` (segundos de la llamada) **se excluye**: es fuga de información.
- **Fuente/cita**: Moro, S., Cortez, P. & Rita, P. (2014), Decision Support Systems 62, 22–31. UCI, CC BY 4.0.

---

## 2. Conceptos que igual debes manejar en la defensa

### 2.1 Perceptrón
Una neurona: `salida = σ(w·x + b)`. Aprende una **frontera lineal**. Aquí se implementó desde cero (NumPy) como **línea base** = regresión logística. Su gradiente con BCE+sigmoide es `(1/N)·Xᵀ(σ(Xw+b) − y)`.

### 2.2 Perceptrón multicapa (MLP)
Varias capas con **activación no lineal** entre medio → puede aprender fronteras **no lineales** e interacciones entre variables. Arquitectura del notebook:

```
Entrada (~51 feat.) → [Linear → (BatchNorm) → Activación → (Dropout)] × k → Linear(1) → logit → sigmoid → p
```

- La capa oculta de 64 → 32 neuronas es el **embudo** base.
- **Aprendizaje**: forward → pérdida → backpropagation (regla de la cadena) → gradiente → optimizador actualiza pesos.

### 2.3 Activaciones y derivadas (para poder explicar convergencia)

| Función | Rango | Derivada máx. | Punto clave |
|---|---|---|---|
| Sigmoide | (0,1) | 0,25 | Se satura → desvanecimiento del gradiente |
| Tanh | (−1,1) | 1 | Centrada en 0, mejor que sigmoide, se satura |
| ReLU | [0,∞) | 1 | Gradiente estable, barata; riesgo neuronas muertas |
| Leaky ReLU | ℝ | 1 / α | Evita neuronas muertas |
| ELU | (−α,∞) | 1 / α·e^z | Suave, media ≈ 0 |
| Softmax | (0,1) suma 1 | — | Multiclase; con 2 salidas equivale a sigmoide |

**Mensaje clave**: la derivada máx. de la sigmoide (0,25) multiplicada en cada capa **apaga el gradiente** en redes profundas. Lo demostramos midiendo el gradiente que llega a la primera capa (gráfico en escala log): con sigmoide es **órdenes de magnitud menor** que con ReLU.

### 2.4 Funciones de pérdida
- **BCE**: `L = −[y·log p + (1−y)·log(1−p)]`. Pérdida de máxima verosimilitud para Bernoulli. Con sigmoide, gradiente respecto al logit = `p − y` (no se satura).
- **BCE ponderada**: `pos_weight ≈ 7,5` (neg/pos). Cada positivo pesa como 7,5 negativos → compensa desbalance.
- **Focal loss**: `L = −α(1−p_t)^γ·log p_t`. Reduce el peso de ejemplos fáciles, enfoca a los difíciles.
- **MSE**: para regresión. Con sigmoide, gradiente `2(p−y)·p(1−p)` → se **anula cuando la neurona se satura aunque esté equivocada**. Por eso es la peor para este problema (lo mostramos con el gráfico de |dL/dz|).

### 2.5 Salida
- Red binaria → **una neurona de salida sin activación (logit)** + `BCEWithLogitsLoss` (numéricamente estable, truco log-sum-exp).
- La sigmoide se aplica después para obtener **probabilidad de contratar** (lo que el banco necesita para priorizar).

### 2.6 Optimizadores
- **SGD**: lento sin momentum. **SGD + momentum (Nesterov)**: acelera.
- **RMSprop / Adam / AdamW**: adaptativos, ajustan el paso por parámetro según historial del gradiente → convergen rápido. Útil con one-hot (muchas columnas casi siempre 0).
- Notebook elige **Adam** (rápido, robusto, no hay que afinar lr).

### 2.7 Regularización
- **Dropout (p=0.3)**: apaga neuronas al azar por paso → ensamble implícito, no depende de neuronas concretas. Va **después de la activación**.
- **BatchNorm**: normaliza activaciones por mini-batch → gradientes mejor condicionados, convergencia más estable. Va **antes de la activación** (Ioffe & Szegedy 2015).
- **L2 / weight decay (1e-4)**: penaliza pesos grandes → funciones más suaves.
- **Early stopping**: si la pérdida de validación no mejora en 8 épocas, corta y **restaura los mejores pesos**. Rojo final de seguridad.
- **Scheduler `ReduceLROnPlateau`**: baja el lr a la mitad cuando la validación se estanca (patience 3).

---

## 3. Preprocesamiento (¿por qué cada paso?)

Todas las decisiones salen de un **EDA** previo (secciones 3 y 4 del notebook).

| Paso | Qué | Por qué |
|---|---|---|
| 1 | Eliminar `duration` | Fuga de información; se conoce **después** de la llamada |
| 2 | `y` → 1/0 | La red y la pérdida usan números |
| 3 | `contactado_antes` y `pdays=-1 → 0` | `-1` es un código ("nunca contactado", 82 %), no un número |
| 4 | `sign(x)·log(1+|x|)` en balance, campaign, pdays, previous | Reduce asimetría/outliers; `StandardScaler` solo aplastaría todo a ~0 |
| 5 | Partición estratificada **70/15/15** | Mantener ~11,7 % de positivos en cada conjunto |
| 6 | `StandardScaler` (fit solo en train) | Escalas comparables → convergencia; sin fuga |
| 7 | `OneHotEncoder(handle_unknown="ignore")` (fit solo en train) | Categóricas nominales sin orden; "unknown" queda como categoría |
| 8 | Tensores `float32` | Formato de PyTorch |

Detalles para defender:
- **"unknown" se trata como categoría propia** (no se borran filas): en `poutcome` es 82 % y en `contact` 29 %; borrar destruiría el set, y "no conocer el dato" en sí es informativo.
- **Fuga (data leakage)**: escalador y one-hot se ajustan **solo con train**; validación y test solo se transforman. En la CV, cada fold reajusta su propio preprocesador.
- Entrada final ≈ **51 features** (6 numéricas + 1 binaria + 44 one-hot).

---

## 4. Experimentos controlados (lo que se probó y qué se interpreta)

**Metodología**: en cada experimento solo cambia **un parámetro**, misma semilla (42), mismos datos, sin early stopping salvo que se indique. Se mira pérdida de validación (convergencia/sobreajuste) y PR-AUC (calidad del ranking).

### 4.1 Épocas
- Se entrena 100 épocas: train baja siempre, **val llega a un mínimo temprano y luego sube** → sobreajuste (el modelo memoriza).
- **Decisión**: early stopping (paciencia 8) en vez de fijar épocas a mano.

### 4.2 Tasa de aprendizaje
- `1e-4`: lento, no termina de converger.
- `1e-3`: **el mejor equilibrio** (suave, alcanza mínimo, luego sobreajusta).
- `1e-2` / `5e-2`: el mínimo llega rápido pero val **sube rápido y ruidoso** (pasos grandes).
- **Decisión**: lr = 1e-3 + scheduler.

### 4.3 Batch size
- 32: actualiza mucho → aprende rápido pero sobreajusta antes y es el más lento.
- 2048: curvas suaves pero convergencia lenta en 40 épocas.
- 128–512: equilibrio. **Decisión: 256** (y se anota que batch y lr interactúan: batch grande admite lr mayor).

### 4.4 Funciones de activación (sección 8)
- Red 64→32: ReLU/Leaky ReLU/ELU casi iguales; tanh algo más lenta; **sigmoide la más lenta**.
- Red profunda 5×32: se amplía la diferencia → gradiente en capa 1 con sigmoide es **mucho menor** (desvanecimiento).
- **Decisión: ReLU** en ocultas; sigmoide solo en salida (probabilidad).

### 4.5 Funciones de pérdida (sección 9)
- **BCE**: buen ranking, pero probabilidades concentradas < 0,5 → precision alta, recall bajo.
- **BCE ponderada**: sube las probabilidades → **recall mucho mayor**, mejor F1 a umbral 0,5. El PR-AUC ≈ igual a BCE: la ponderación mueve el umbral, no el ranking.
- **Focal**: también mejora recall.
- **MSE**: la peor (gradiente que se anula con sigmoide saturada).
- **Decisión: BCE ponderada** (máxima verosimilitud, estable, y el negocio valora más no perder a un cliente que haría una llamada extra).

### 4.6 Optimización y regularización (sección 10)
- Sin regularización: brecha val−train crece, PR-AUC cae (sobreajuste).
- **Dropout: la técnica individual más efectiva** (val estable, brecha baja, mantiene PR-AUC).
- L2: efecto moderado. BatchNorm: acelera pero no regulariza suficiente.
- **Combinación Dropout+BN+L2**: curvas más estables, menor brecha.
- **Decisión final**: Dropout + L2 + early stopping (BatchNorm se evalúa en la grilla).

### 4.7 Arquitectura e hiperparámetros (sección 11)
- Se prueban 7 arquitecturas (16 → hasta 4×64): sobre 2 capas las diferencias son chicas → problema tabular, moderada dimensión.
- **Regla de parsimonia**: entre las arquitecturas a menos de 0.003 de PR-AUC de la mejor, elige la de **menos parámetros**.
- Grilla: activación × lr × dropout × batch × batchnorm (32 combinaciones, evaluadas en validación).
- Se reporta el **efecto marginal** de cada hiperparámetro (gráficos de caja). Diferencias pequeñas → el modelo es robusto.
- `mejor_cfg` + scheduler = configuración final.

### 4.8 Modelo final, umbral y test (sección 12)
- Umbral: en **validación** se busca el que maximiza **F1** (`precision_recall_curve`). Hacerlo en test sería sobreestimación.
- Test se usa **una sola vez**, como referencia: trivial ("siempre no"), regresión logística (scikit-learn, `class_weight="balanced"`), MLP base y MLP final.
- Se entrega cuadro resumen de métricas en test, matriz de confusión, curvas ROC y PR, y `classification_report`.
- Se calculan las métricas **a mano** desde la matriz de confusión y se verifican contra sklearn.

---

## 5. Métricas (cómo explicarlas con ejemplo)

| Métrica | Fórmula | Significado | Defensa |
|---|---|---|---|
| Accuracy | (TP+TN)/N | fracción bien clasificada | **Engañosa**: el trivial ya da ~88 % |
| Precision | TP/(TP+FP) | de los llamados, cuántos contratan | eficiencia del call center |
| Recall | TP/(TP+FN) | de los que habrían contratado, a cuántos llegamos | oportunidades capturadas |
| F1 | 2·P·R/(P+R) | media armónica; alta solo si ambas lo son | métrica principal del informe |
| ROC-AUC | área bajo ROC | ranking global | menos sensible al desbalance |
| PR-AUC (AP) | área bajo P-R | ranking de la clase minoritaria | **la que guía los experimentos** |

Interpretación para defender:
- El 88 % de accuracy del trivial no sirve: **lo que importa es precision/recall/F1 de la clase "yes"**.
- El umbral es **una palanca de negocio**: bajar umbral → más recall, menos precision (más llamadas fallidas). El "mejor" umbral depende del costo para el banco. Se muestra el gráfico de qué se gana/pierde según el umbral en test.
- Con el umbral óptimo, el MLP final concentra sus llamadas en un subconjunto de clientes **con tasa de conversión varias veces mayor** que llamar al azar (~11 % de conversión al azar).

---

## 6. Validación cruzada (sección 13)

- **5 folds estratificados** sobre train+val (test intacto).
- En cada fold: preprocesador ajustado solo con el train del fold, validación interna (15 %) para early stopping, umbral elegido en esa validación.
- Si la desviación estándar entre folds es pequeña → el resultado **no depende de una partición afortunada**.

---

## 7. Análisis crítico y propuestas de mejora (sección 14)

**Interpretación (que debes poder decir con palabras):**
1. Accuracy ~0,88–0,90 no es informativa; el valor está en precision/recall/F1.
2. El MLP supera claramente al azar y al perceptrón lineal; la ventaja sobre regresión logística es **moderada**: en tabulares buena parte de la señal es ~lineal tras el one-hot; la red aporta en interacciones.
3. El techo lo pone la información disponible, no la red (por eso más capas no ayudaban).
4. Las mejoras de más impacto: **desbalance** (pérdida ponderada + umbral) y **sobreajuste** (dropout/L2/early stopping), más que arquitectura.

**Propuestas fundamentadas:**
- Ingeniería de variables: codificación cíclica de month/day (sin y cos), interacciones (poutcome × contactado_antes), agrupar categorías raras.
- **Embeddings** (`nn.Embedding`) para categóricas en vez de one-hot.
- Datos adicionales: `bank-additional` de UCI tiene **macroeconómicos** (euribor, empleo), muy predictivos según la literatura.
- SMOTE/oversampling como alternativa a la ponderación.
- Búsqueda más amplia (Optuna), ensemble o ajuste de un **umbral óptimo por costo**.

**Evidencia de la fuga de `duration`**: se entrena la misma configuración incluyendo `duration` → las métricas suben drásticamente. Confirma que "luce mejor pero es inutilizable" (revela el futuro). Esto demuestra comprensión profunda.

---

## 8. Estructura propuesta para la presentación (10 min)

1. **Introducción (1 min)**: problema, dataset, objetivo, métrica de éxito.
2. **Carga y preprocesamiento (2 min)**: EDA → decisiones (1–2 ejemplos). Mostrar código funcionando.
3. **Arquitectura del MLP (1 min)**: clases, activación, salida logit + sigmoid, init He/Xavier.
4. **Funciones implementadas desde cero (1,5 min)**: tabla de activaciones/derivadas, verificación vs PyTorch, BCE vs MSE (gráfico de gradiente).
5. **Experimentos (2 min)**: un gráfico bueno de cada experimento (lr, batch, épocas; activación; pérdida; regularización; grid). "Variamos 1 parámetro a la vez, misma semilla".
6. **Métricas y umbral (1,5 min)**: explicar cada métrica con el ejemplo de la matriz de confusión, umbral óptimo, cuadro en test.
7. **Conclusiones y mejoras (1 min)**: 3 mensajes + 2 propuestas.

> El docente valora gráficos **bien elaborados y relevantes** que respalden lo que dices, y que **domines el código** (puede pedirte ejecutar celdas en vivo).

---

## 9. Preguntas probables del docente (con respuesta corta)

1. **¿Por qué excluyes `duration`?** → Fuga: se conoce solo después de llamar y la llamada termina cuando acepta/rechaza. Un modelo con ella no sirve para decidir a quién llamar. Lo demostramos cuantitativamente.
2. **¿Por qué no usas accuracy como métrica principal?** → Desbalance ~11,7 %: el clasificador trivial ya da ~88 %. Usamos F1/precision/recall y PR-AUC.
3. **¿Qué es early stopping y por qué restauras los pesos?** → Es el mejor punto en validación; restaurar los mejores pesos evita quedarse con los pesos sobreajustados del último paso.
4. **¿Por qué BCE y no MSE?** → MSE con sigmoide anula el gradiente cuando se satura (aunque se equivoque); BCE da gradiente máximo (p−y) y no se satura. Mostrar gráfico.
5. **¿Una neurona de salida o softmax?** → Con 2 clases, sigmoide es equivalente a softmax (lo verificamos). Usamos 1 neurona con logit + sigmoide (menos parámetros, estable).
6. **¿Por qué ReLU en ocultas?** → No se satura, gradiente estable, barata; en este problema igual o mejor que las otras. La sigmoide sufre desvanecimiento (evidencia del gradiente en capa 1).
7. **¿Dropout y BatchNorm dónde van?** → BN antes de la activación (normaliza el pre-activación); Dropout después de la activación.
8. **¿Qué es pos_weight?** → Ratio neg/pos en train (~7,5). Cada positivo "pesa" como 7,5 negativos: la red castiga más equivocarse en un "yes".
9. **¿Para qué el umbral óptimo?** → La red entrega probabilidad; 0,5 es arbitrario. Buscamos en validación el umbral que maximiza F1. En test sería "ver el resultado dos veces".
10. **¿Por qué estratificar?** → Mantiene la proporción de clases en train/val/test; una partición aleatoria podría dejar un test sin "yes".
11. **¿Cómo evitas fuga en el preprocesamiento?** → Scaler y one-hot se ajustan solo con train; val/test solo se transforman; en cada fold de CV se reajusta.
12. **¿Por qué PR-AUC y no solo ROC-AUC?** → Con mucho desbalance, ROC es optimista; PR-AUC se enfoca en la clase positiva minoritaria.
13. **¿El MLP es mejor que regresión logística?** → Sí, pero ventaja moderada: datos tabulares con señal ~lineal; la red aporta en interacciones no lineales.
14. **¿Cómo elegiste el número de capas/neuronas?** → Búsqueda de arquitecturas + regla de parsimonia (dentro de 0.003 del mejor, elijo la de menos parámetros).
15. **¿Qué hace el scheduler?** → ReduceLROnPlateau baja el lr a la mitad cuando val se estanca: partir rápido y afinar al final.
16. **¿Qué demuestra la validación cruzada?** → Que el desempeño no depende de una partición afortunada (σ pequeña entre folds).

---

## 10. Checklist antes de entregar/defender

- [ ] Rellenar **Integrantes** en la portada del notebook (cambiar "Nombre Apellido 1/2").
- [ ] **Agregar la sección 16 Referencias** (el índice la promete y no existe): Moro et al. 2014, Kingma & Ba 2015 (Adam), Ioffe & Szegedy 2015 (BatchNorm), Lin et al. 2017 (Focal loss), UCI repositorio.
- [ ] Dejar el notebook **corrido** (celdas con salidas visibles) antes de proyectar.
- [ ] Subir a **GitHub** y tener el enlace listo (instrucciones de ejecución en README).
- [ ] Revisar que la fila "Trivial (siempre no)" de la tabla de test no tire error (roc_auc con probabilidades constantes).
- [ ] Memorizar 3 mensajes (los que empezaste con "La lección más importante fue: entender los datos y el negocio (fuga, desbalance, umbral), no agrandar la red").

---

## 11. Nota rápida sobre resultados (tras terminar la ejecución)

Los valores exactos en test/cv se ven al correr todas las celdas. Para la defensa, ubica e interpreta en el notebook:
- **Tabla de test** (sección 12.3): compara trivial vs logreg vs MLP base vs MLP final (umbral 0.5 y óptimo).
- **Matriz de confusión** (12.4): anota los 4 números (TP/FP/FN/TN) y repasa el cálculo manual.
- **Tabla CV** (sección 13): media y desviación estándar de F1/PR-AUC por fold.
- **Curvas de entrenamiento del modelo final** (12.1): train vs val + lr (scheduler).
- **Gráfico umbral** (14.2): cuántos clientes llamarías y qué precision/recall obtienes en cada umbral.