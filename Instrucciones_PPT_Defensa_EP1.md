# Instrucciones PPT — Defensa EP1 MLP Bank Marketing

> Guía detallada, diapositiva por diapositiva, para construir la presentación (PPT) de la **defensa oral** de la Evaluación Parcial 1 (DLY0100 · Deep Learning).
> **Formato Básico:** Celda `me di cuenta que las celdas están truncadas por el tool` no aplica.
> **Tiempo:** 10 minutos en dupla → ~5 min por integrante. **Ponderación:** 80 % de la nota del encargo.

---

## 0. Reglas de oro antes de empezar

1. **Máximo 12–14 diapositivas** para 10 min. Una idea por slide. Si una diapositiva tiene más de 6 viñetas, se divide.
2. **Todo gráfico salta del notebook** (celdas indicadas más abajo) con sus **salidas visibles** (cortar la captura con el resultado impreso). El docente valora gráficos *bien elaborados y relevantes*, no solo bonitos: cada figura debe **respadlar una afirmación** que dices.
3. **Cada afirmación técnica lleva evidencia** (dato, gráfico o tabla). Nada de opiniones sin respaldo.
4. **Texto en las diapositivas: mínimo.** Las viñetas son apoyo; lo que "se dice" está en las notas del orador.
5. Los nombres de los integrantes deben aparecer (el header del notebook aún los tiene como plantilla).
6. Proyectar **el notebook corriendo** con las celdas ejecutadas y pasar a PPT solo para pantallas clave — el indicador IP 1.3.1 premia "ejecutar el código en vivo".

---

## 1. Mapa diapositivas → rúbrica (cobertura 100 % de la presentación)

| # | Diapositiva | Tiempo | Indicador que cubre |
|---|---|---|---|
| 1 | Portada | 0:20 | — |
| 2 | Problema y objetivo | 0:40 | — |
| 3 | Datos y hallazgos del EDA | 0:50 | IP 1.1.1 (preludio) |
| 4 | Preprocesamiento y decisiones | 1:30 | **IP 1.1.1** |
| 5 | Arquitectura MLP | 1:00 | **IP 1.3.1** |
| 6 | Funciones programadas desde cero | 1:30 | **IP 1.2.1** |
| 7 | Comparación de funciones (act/pérdida) | 1:00 | **IP 1.2.2, 1.2.3** |
| 8 | Experimentos: épocas · lr · batch | 1:30 | **IP 1.1.2** |
| 9 | Optimización y regularización | 1:00 | **IP 1.3.2** |
| 10 | Ajuste de hiperparámetros | 1:00 | **IP 1.3.3** |
| 11 | Evaluación en test + cuadro resumen | 1:30 | **IP 1.4.1, 1.4.2** |
| 12 | Umbral de decisión (negocio) | 0:50 | **IP 1.4.2** |
| 13 | Validación cruzada + conclusiones + mejoras | 1:00 | **IP 1.1.3, 1.4.3** |
| 14 | Preguntas / gracias | 0:20 | — |

**Total ≈ 14:00 bruto → recortar a 10:00.** Prioridad de recorte: fusionar 9+10, o quitar la 3 y pasar su contenido a la 4.

---

## 2. Ficha por diapositiva

### Diapositiva 1 — Portada
- **Contenido:** Título "Red Neuronal Multicapa para clasificar la contratación de un depósito a plazo (Bank Marketing)"; asignatura **DLY0100 – Deep Learning**; **EP1**; integrantes; fecha.
- **Visual:** logo de la institución (opcional), nothing más.
- **Guion:** "Buenas, somos [Nombres]. Vamos a presentar la implementación de un MLP en PyTorch para predecir si un cliente contratará un depósito a plazo."
- **Tiempo:** 0:20.

### Diapositiva 2 — Problema y objetivo
- **Contenido (5 viñetas máx.):**
  - Banco portugués → campaña de **marketing telefónico** (dataset UCI *Bank Marketing*, Moro et al. 2014).
  - **Objetivo:** predecir si el cliente contratará el depósito (`y = yes/no`) → decidir **a quién llamar**.
  - Clasificación **binaria** · 45.211 clientes · 16 variables (7 numéricas + 9 categóricas).
  - Problema **desbalanceado**: solo ~11,7 % son "yes".
  - **Métrica de éxito:** F1 / precision / recall de la clase "yes" (+ PR-AUC). Accuracy no sirve.
- **Visual:** captura de la tabla de contexto (celda c001 del notebook) o la tabla resumen inicial.
- **Guion:** explicar en una frase para qué sirve el modelo y por qué importa (priorizar llamadas del call center). Mencionar que la fuente es UCI y citar el paper.
- **Tiempo:** 0:40.

### Diapositiva 3 — Datos y hallazgos del EDA
- **Contenido:**
  - Desbalance fuerte: *por cada "yes" hay ~7,5 "no"* (gráfico de barras de `y`).
  - `"unknown"` muy frecuente en `poutcome` (82 %) y `contact` (29 %) → no se pueden borrar filas.
  - Numéricas con **asimetría alta** (`balance`, `campaign`, `previous`, `pdays`) → necesitan transformación log.
  - `pdays = -1` es un **código** ("nunca contactado"), no un número.
  - `duration` separa las clases pero es **fuga de información** (caja: duración vs resultado).
- **Visuales:** c012 (barras `y`), c018 (histogramas), c024 (boxplot duration).
- **Guion:** "Cada hallazgo del EDA motivó una decisión de preprocesamiento. Los vemos ahora."
- **Tiempo:** 0:50.

### Diapositiva 4 — Preprocesamiento y justificación  ⭐ IP 1.1.1
- **Contenido (tabla):**
  | Decisión | Por qué |
  |---|---|
  | Excluir `duration` | Fuga: se conoce solo después de la llamada |
  | `pdays=-1` → `contactado_antes` binaria | Es código, no cantidad |
  | `sign(x)·log(1+|x|)` en 4 numéricas | Reduce asimetría y outliers |
  | Partición estratificada 70/15/15 | Mantener ~11,7 % de "yes" en cada conjunto |
  | `StandardScaler` + `OneHotEncoder` (fit **solo en train**) | Escalas/ordinalidad; evita **data leakage** |
  | `"unknown"` = categoría nueva | No borrar filas; no-conocer-el-dato es informativo |

- **Visuales:** c026 (tabla resumen de decisiones), c027/c028 (código de partición y ColumnTransformer con salida), c029 (histograma campaign: original → log → log+scaler).
- **Guion clave:** explicar **fuga de datos** con palabras: "ajustamos el escalador y el one-hot solo con train; val y test solo se transforman. En la validación cruzada reajustamos por fold." Esta diapositiva se defiende con código en vivo si preguntan.
- **Tiempo:** 1:30.

### Diapositiva 5 — Arquitectura MLP  ⭐ IP 1.3.1
- **Contenido:**
  - Diagrama: `Entrada (~51 feat) → [Linear → (BatchNorm) → Activación → (Dropout)] × capas → Linear(1) → logit → sigmoid → p`.
  - **1 neurona de salida sin activación (logit)** + `BCEWithLogitsLoss` (estabilidad numérica, truco log-sum-exp).
  - Inicialización: **He/Kaiming** (ReLU) · **Xavier** (tanh/sigmoide).
  - **BatchNorm antes** de la activación · **Dropout después** (Ioffe & Szegedy 2015).
  - Arquitectura **parametrizable** → permite los experimentos de las siguientes diapositivas.
- **Visuales:** c038 (print del modelo + n° de parámetros con salida).
- **Guion:** "La red es un embudo de capas totalmente conectadas. Elegimos salida logit + sigmoide, y todo se parametriza para poder experimentar." En vivo: mostrar la clase `MLP` en la celda.
- **Tiempo:** 1:00.

### Diapositiva 6 — Funciones programadas desde cero  ⭐ IP 1.2.1
- **Contenido:**
  - Activaciones implementadas en NumPy **con sus derivadas**: sigmoide, tanh, ReLU, Leaky ReLU, ELU, softmax.
  - Pérdidas: BCE, BCE ponderada, focal, MSE (plus sigmoide).
  - **Validación contra PyTorch:** error máximo ≈ 1e-16 (precisión de máquina).
  - Softmax(2) ≡ sigmoide → por eso 1 sola neurona de salida.
  - **BCE vs MSE:** gradiente `p−y` (no satura) vs `2(p−y)p(1−p)` (se anula cuando la sigmoide se satura aunque esté equivocada).
- **Visuales:** c031 (gráfico f(z) y f'(z)), c032 (tabla de verificación), c034 (curvas de pérdida y |gradiente|).
- **Guion clave:** "Programamos activaciones, sus derivadas y las pérdidas desde cero, y las validamos contra PyTorch. Esto es lo que determina cómo fluye el gradiente en el backpropagation."
- **Tiempo:** 1:30.

### Diapositiva 7 — Comparación de funciones  ⭐ IP 1.2.2, IP 1.2.3
- **Contenido:**
  - Activaciones en red 64→32 **y en red profunda 5×32**: sigmoide sufre **desvanecimiento del gradiente** (barra log del gradiente en capa 1, c052).
  - Pérdidas (early stopping): **BCE ponderada** gana en recall/F1; MSE la peor.
  - **Decisión:** ReLU en ocultas (rápida, estable), sigmoide en salida (probabilidad), BCE ponderada como pérdida (máxima verosimilitud + desbalance 1:7,5).
- **Visuales:** c051 + c052 (comparación activación, gradiente capa 1), c055 (pérdidas), c056 (histogramas de probabilidad predicha según pérdida).
- **Guion:** señalar la tabla comparativa con highlight verde del mejor, y decir: "cada función se eligió según lo que pedía el caso: minimizar falsos negativos (perder un cliente que sí contrataría) es más caro que una llamada extra."
- **Tiempo:** 1:00.

### Diapositiva 8 — Experimentos: épocas · lr · batch  ⭐ IP 1.1.2
- **Contenido (3 mini-bloques con gráfico cada uno):**
  - **Épocas:** 100 épocas → train baja, **val sube después del mínimo** → sobreajuste → **early stopping** (paciencia 8, restaura mejores pesos). Gráfico c044.
  - **lr:** 1e-4 no converge; 1e-3 equilibrado; 1e-2/5e-2 sobreajustan rápido y ruidoso → **1e-3**. Gráfico c046.
  - **batch:** 32 ruidoso/lento; 2048 lento; 256 equilibrio → **256**. Gráfico c048.
- **Contenido metodología:** *"variamos un parámetro a la vez, misma semilla (42), mismos datos"* → los resultados solo dependen del parámetro estudiado.
- **Visuales:** c044, c046, c048 (curvas de train/val + PR-AUC).
- **Guion clave:** "Aquí la clave metodológica: experimentos controlados, un parámetro por vez. Y las tablas muestran tanto el valor final como el mejor punto de cada configuración, para separar velocidad de convergencia y estabilidad."
- **Tiempo:** 1:30.

### Diapositiva 9 — Optimización y regularización  ⭐ IP 1.3.2
- **Contenido:**
  - Sin regularización: **brecha val−train crece** y PR-AUC cae → sobreajuste.
  - **Dropout 0.3**: técnica individual más efectiva (val estable, brecha baja).
  - L2 (1e-4): efecto moderado. **BatchNorm**: acelera, pero no regulariza sola.
  - Combinación **Dropout+BN+L2**: curvas más estables, menor brecha, menor variación del PR-AUC.
  - Optimizadores: SGD lento → momentum acelera → **Adam** (adaptativo) elige.
- **Visuales:** c059 (3 paneles: pérdida train/val, brecha, PR-AUC por variante), c061 (optimizadores).
- **Guion:** "Medimos regularización con la brecha de generalización (val − train) y con la desviación del PR-AUC; la combinación es la más estable."
- **Tiempo:** 1:00.

### Diapositiva 10 — Ajuste de hiperparámetros  ⭐ IP 1.3.3
- **Contenido:**
  - Arquitectura: se compararon 7; por **parsimonia** se elige la de menos parámetros dentro de 0.003 del mejor PR-AUC (gráfico c064).
  - **Grilla** (32 combinaciones): activación × lr × dropout × batch × batchnorm, evaluada en **validación** (nunca test).
  - **Efecto marginal** de cada hiperparámetro (gráfico de cajas, c067) → el modelo es robusto.
  - Configuración final impresa (c068): `capas, activación, lr, dropout, batch, batchnorm, scheduler=True`.
- **Visuales:** c064 (PR-AUC por arquitectura), c067 (cajas de efecto marginal), c068 (print config final).
- **Guion:** "Justificamos cada valor con la combinación de mayor PR-AUC en validación y añadimos un scheduler ReduceLROnPlateau para afinar el lr al final."
- **Tiempo:** 1:00.

### Diapositiva 11 — Evaluación en test y cuadro resumen  ⭐ IP 1.4.1, IP 1.4.2
- **Contenido:**
  - Cuadro resumen en **test** (usado una sola vez): Trivial ("siempre no") vs Regresión logística vs MLP base vs **MLP final** (umbral 0.5 y óptimo).
  - Explicar **cada métrica** con la fórmula y un ejemplo (Matriz de confusión TP/FP/FN/TN, c076/c078).
  - Curvas **ROC y Precision-Recall** en test para los 3 modelos.
  - Mensaje: valor real está en precision/recall/F1 de "yes", no en accuracy (~0.88 igual al trivial).
- **Visuales:** c075 (cuadro resumen con highlight), c076 (confusión + ROC/PR), c078 (cálculo manual impreso).
- **Guion clave:** "Las métricas se calcularon a mano desde la matriz de confusión y se verificaron contra scikit-learn. Eso es lo que vamos a defender." → Al detalle: TP, FP, FN, TN y fórmulas.
- **Tiempo:** 1:30.

### Diapositiva 12 — Umbral de decisión (palanca de negocio)  ⭐ IP 1.4.2
- **Contenido:**
  - La red entrega **probabilidad**; 0,5 es arbitrario → se elige en **validación** el umbral que **maximiza F1**.
  - Gráfico Precision/Recall/F1 vs umbral (c073) y efecto en test (c086): qué % de clientes se llama y qué se gana/pierde.
  - Interpretación: bajar umbral ↑ recall, ↓ precision (más llamadas fallidas) — depende del costo del banco.
- **Visuales:** c073 (trade-off del umbral), c086 (curvas en test por umbral).
- **Guion:** "Elegir el umbral en test habría sido 'ver el resultado dos veces'; por eso se optimiza en validación. El umbral es una palanca operativa para el call center."
- **Tiempo:** 0:50.

### Diapositiva 13 — Validación cruzada + conclusiones + mejoras  ⭐ IP 1.1.3, IP 1.4.3
- **Contenido:**
  - **5-fold estratificado** sobre train+val (test intacto), preprocesador reajustado por fold → σ pequeña entre folds = resultado **confiable** (c080).
  - **Evolución incremental** de métricas por etapa de mejora (c084: perceptrón → MLP base → +BCE ponderada → +regularización → +tuning → +umbral).
  - Conclusiones: el mayor aporte fue **entender datos y negocio** (fuga, desbalance, umbral), no agrandar la red.
  - **Mejoras futuras:** embeddings para categóricas, variables macroeconómicas (bank-additional), SMOTE, codificación cíclica de mes.
- **Visuales:** c080 (tabla CV con media y std), c084 (líneas de evolución).
- **Guion:** cerrar con 3 frases: (1) pipeline sin fuga y validado, (2) desbalance resuelto con pérdida ponderada + umbral, (3) CV confirma estabilidad. "Lo aprendido: la red aporta en interacciones, pero el techo lo pone la información disponible."
- **Tiempo:** 1:00.

### Diapositiva 14 — Cierre
- **Contenido:** "¿Preguntas?" + datos de contacto/integrantes.
- **Tiempo:** 0:20.

---

## 3. Checklist de armado (por integrante)

- [ ] Descargar capturas del notebook **terminado de correr** (con números reales en las tablas).
- [ ] Las capturas incluyen los **títulos/labels legibles** (dpi del notebook = 100; usar zoom si hace falta).
- [ ] Reemplazar el placeholder de integrantes en **portada** y en el **notebook**.
- [ ] Colores coherentes (aprovechar el tema `deep`/verde de highlight ya usado).
- [ ] Notas de orador escritas (este mismo guion) en el PPT; ensayar con cronómetro.
- [ ] Dividir el habla: dupla = ~5 min c/u, por ejemplo A: slides 2–6, B: 7–12, cierran juntos en 13–14.
- [ ] Tener el notebook abierto y corrido como plan B/ejecución en vivo.
- [ ] ENSAYO: grabarse una vez; verificar que las transiciones y los gráficos refuercen (no distraigan).

## 4. Errores comunes a evitar en la defensa

1. Decir "el modelo tiene 88 % de accuracy, ¡muy bueno!" → el docente pisará eso (el trivial también). Siempre aterrizar en F1/precision/recall.
2. No poder responder **por qué se quitó `duration`** → es la pregunta estrella.
3. Confundir **precision** y **recall** en las preguntas.
4. Decir "BatchNorm reduce overfitting" como verdad absoluta → aquí aceleró convergencia pero no regularizó sola; lo cuentas con la evidencia de la brecha.
5. Leer las diapositivas → el docente evalúa **dominio del notebook** y respuestas.
6. No saber qué cambiaste **después** de ver resultados (quién: early stopping, pérdida ponderada, umbral, dropout/L2) — IP 1.1.3 e IP 1.3.3.

## 5. Frases comodín para preguntas difíciles

- **"¿El MLP es mejor que regresión logística?"** → "En este problema sí, pero la ventaja es moderada: en datos tabulares gran parte de la señal es aproximadamente lineal tras el one-hot; la red aporta sobre todo en interacciones no lineales."
- **"¿Por qué PR-AUC y no solo ROC?"** → "Con desbalance fuerte el ROC es optimista porque da igual peso a las dos clases; el PR-AUC se enfoca en la clase positiva minoritaria, que es la que nos interesa."
- **"¿Y si uso softmax?"** → "Con dos clases softmax es equivalente a una sigmoide sobre la diferencia de logits; lo verificamos en el notebook. Una neurona con sigmoide es lo mismo con menos parámetros."
- **"¿Cómo sabes que no hay fuga?"** → "El escalador y el one-hot se ajustan solo con train; en la CV cada fold reajusta el suyo; y test se usa una sola vez, al final."
- **"¿Cuál fue el cambio más importante?"** → "Atacar el desbalance (BCE ponderada + umbral) y el sobreajuste (dropout/L2 + early stopping) importó mucho más que la arquitectura."