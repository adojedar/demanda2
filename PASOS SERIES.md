

# Guía Completa para Generación de Pronósticos: Estadística Clásica y Machine Learning

Te voy a crear una guía exhaustiva paso a paso. Empezaré con un diagrama de flujo general y luego profundizaré en cada etapa.

Ahora te detallo cada fase:

## **FASE 1: ANÁLISIS EXPLORATORIO Y PREPARACIÓN**

### 1.1 Inspección Inicial de Datos

**Pasos:**
- Verificar estructura temporal: frecuencia, gaps, continuidad
- Identificar el rango temporal y granularidad
- Detectar cambios de régimen o puntos de quiebre estructural

**Visualizaciones clave:**
- Gráfico de línea temporal
- Histograma de la variable objetivo
- Box plot por período (mes, trimestre, año)

### 1.2 Tratamiento de Valores Faltantes

**Métodos según el caso:**

1. **Interpolación lineal**: Para gaps pequeños (<5% de datos)
2. **Interpolación spline**: Para series suaves
3. **Forward/Backward fill**: Para valores categóricos o cuando el último valor es representativo
4. **Interpolación estacional**: Cuando hay estacionalidad clara
5. **Modelos de imputación**: KNN, MICE, algoritmos ML para gaps grandes
6. **Eliminación**: Solo si es <1% y aleatorio

**Test para evaluar aleatoriedad de missings:**
- Test de Little's MCAR
- Visualización de patrones de missing con heatmaps

### 1.3 Detección y Tratamiento de Outliers

**Métodos de detección:**

1. **Estadísticos:**
   - Z-score (|z| > 3)
   - IQR: Q1 - 1.5×IQR, Q3 + 1.5×IQR
   - Modified Z-score (MAD)

2. **Modelos:**
   - Isolation Forest
   - Local Outlier Factor (LOF)
   - DBSCAN

3. **Para series temporales:**
   - STL decomposition + análisis de residuos
   - Twitter's AnomalyDetection
   - Detección de cambios de nivel con PELT

**Tratamiento:**
- **Winsorización**: Limitar a percentiles (1%, 99%)
- **Transformación**: Log, Box-Cox
- **Imputación**: Reemplazar con mediana móvil
- **Mantener**: Si son eventos reales importantes (Black Friday, pandemias)

---

## **FASE 2: TESTS DE ESTACIONARIEDAD**

### 2.1 Tests Formales

**1. Augmented Dickey-Fuller (ADF)**
```
H0: Existe raíz unitaria (NO estacionaria)
H1: Es estacionaria
Rechazar H0 si p-value < 0.05
```

**Configuración:**
- Probar con: 'c' (constante), 'ct' (constante + tendencia), 'n' (ninguna)
- Lags automáticos: AIC, BIC
- Interpretación: ADF statistic < valores críticos

**2. KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**
```
H0: Es estacionaria
H1: NO es estacionaria
Rechazar H0 si p-value < 0.05
```

**Importante:** Es complementario al ADF
- ADF y KPSS estacionaria → CONFIRMAR estacionaria
- ADF estacionaria, KPSS no → Diferencia-estacionaria
- ADF no estacionaria, KPSS estacionaria → Análisis adicional
- Ambos no estacionaria → NO estacionaria

**3. Phillips-Perron (PP)**
- Similar al ADF pero robusto a heteroscedasticidad
- Útil cuando hay cambios de volatilidad

**4. Zivot-Andrews**
- Detecta puntos de quiebre estructural
- Útil para series con cambios de régimen

### 2.2 Inspección Visual

**Gráficos:**
1. **Plot temporal**: Buscar tendencia clara, varianza creciente
2. **Rolling mean y rolling std**: Deben ser constantes
3. **Distribución por períodos**: Comparar distribuciones en distintos rangos temporales

### 2.3 Transformaciones para Estacionariedad

**Decisión por tipo de no-estacionariedad:**

| Problema | Transformación | Cuándo usar |
|----------|---------------|-------------|
| Tendencia determinística | Detrending (regresión) | Tendencia lineal clara |
| Tendencia estocástica | Diferenciación (d=1) | Test ADF indica raíz unitaria |
| Varianza creciente | Log, Box-Cox | Heteroscedasticidad, valores positivos |
| Estacionalidad | Diferenciación estacional (D=1) | Patrones repetitivos |
| Múltiples problemas | Combinación | Log + diferencia + diferencia estacional |

**Box-Cox óptimo:**
- λ = 0: log(y)
- λ = 0.5: √y
- λ = -1: 1/y
- Estimar λ con maximum likelihood

---

## **FASE 3: ANÁLISIS DE AUTOCORRELACIÓN**

### 3.1 ACF (Autocorrelation Function)

**Interpretación:**
- **Decaimiento exponencial**: Proceso AR
- **Corte abrupto en lag q**: Proceso MA(q)
- **Decaimiento lento**: Serie no estacionaria, necesita diferenciación
- **Picos en lags estacionales**: (12, 24, 36 para mensual) → Estacionalidad

### 3.2 PACF (Partial Autocorrelation Function)

**Interpretación:**
- **Corte abrupto en lag p**: Proceso AR(p)
- **Decaimiento exponencial**: Proceso MA
- **Decaimiento sinusoidal**: Combinar AR y MA

### 3.3 Tests de Autocorrelación

**1. Ljung-Box Test**
```
H0: No hay autocorrelación hasta el lag h
H1: Existe autocorrelación
p-value < 0.05 → Rechazar H0
```

**Configuración:**
- lags = min(10, T/5) para no estacional
- lags = min(2m, T/5) para estacional (m = período estacional)

**2. Durbin-Watson**
- Rango: 0 a 4
- DW ≈ 2: No autocorrelación
- DW < 2: Autocorrelación positiva
- DW > 2: Autocorrelación negativa

**3. Breusch-Godfrey**
- Más general que Durbin-Watson
- Detecta autocorrelación de orden superior

---

## **FASE 4: ANÁLISIS DE ESTACIONALIDAD**

### 4.1 Descomposición

**1. Descomposición Aditiva: Y_t = T_t + S_t + R_t**
- Usar cuando amplitud estacional es constante
- Componentes se suman

**2. Descomposición Multiplicativa: Y_t = T_t × S_t × R_t**
- Usar cuando amplitud estacional crece con el nivel
- Transformar a log para convertir en aditiva

**3. STL (Seasonal and Trend decomposition using Loess)**
- Más flexible que descomposición clásica
- Robusto a outliers
- Permite estacionalidad variable
- Parámetros: seasonal (período), trend (ventana de suavizado)

**4. X-13ARIMA-SEATS**
- Estándar de agencias gubernamentales
- Ajuste automático de efectos calendario
- Detección de outliers

### 4.2 Tests de Estacionalidad

**1. Test CH (Canova-Hansen)**
```
H0: Estacionalidad determinística estable
H1: No hay estacionalidad estable
```

**2. Test OCSB (Osborn-Chui-Smith-Birchenhall)**
- Específico para raíces unitarias estacionales
- Útil para series trimestrales

**3. Test de Friedman**
- Test no paramétrico
- Compara medianas entre períodos estacionales

**4. Espectro de Fourier**
- Transformada rápida de Fourier (FFT)
- Identificar frecuencias dominantes
- Picos significativos indican periodicidad

### 4.3 Efectos de Calendario

**Considerar:**
- Días festivos móviles (Semana Santa)
- Días laborables vs fines de semana
- Efectos puente
- Eventos especiales recurrentes
- Diferencia en días del mes/año

---

## **FASE 5: SELECCIÓN DE MODELOS**

### Árbol de Decisión para Modelos

## **MODELOS ESTADÍSTICOS CLÁSICOS**

### 5.1 ARIMA(p,d,q)

**Componentes:**
- **AR(p)**: φ₁y_{t-1} + ... + φₚy_{t-p}
- **I(d)**: Diferenciación d veces
- **MA(q)**: θ₁ε_{t-1} + ... + θ_qε_{t-q}

**Selección de órdenes:**

**Método 1: ACF/PACF**
| ACF | PACF | Modelo |
|-----|------|--------|
| Decae exponencialmente | Corte en lag p | AR(p) |
| Corte en lag q | Decae exponencialmente | MA(q) |
| Decae exponencialmente | Decae exponencialmente | ARMA(p,q) |

**Método 2: Criterios de Información**
- **AIC** = -2log(L) + 2k (penaliza menos)
- **BIC** = -2log(L) + k×log(T) (penaliza más, prefiere modelos simples)
- **AICc** = AIC + 2k(k+1)/(T-k-1) (para muestras pequeñas)

**Método 3: auto.arima (Hyndman-Khandakar)**
```
Algoritmo:
1. Probar 4 modelos base
2. Variar p,q en ±1 del mejor
3. Seleccionar por AICc
4. Criterio de parsimonia: menor complejidad si AICc similar
```

**Rangos razonables:**
- p, q ≤ 5 (raramente mayor)
- d ≤ 2 (casi siempre d=0,1)
- Validar con KPSS después de diferenciación

**Diagnóstico:**
1. Residuos deben ser ruido blanco
2. Test Ljung-Box en residuos (p > 0.05)
3. Test de normalidad: Jarque-Bera, Shapiro-Wilk
4. Heterocedasticidad: ARCH test

### 5.2 SARIMA(p,d,q)(P,D,Q)ₘ

**Componentes adicionales:**
- **P**: Orden AR estacional
- **D**: Diferenciación estacional
- **Q**: Orden MA estacional
- **m**: Período estacional (12 para mensual, 4 para trimestral)

**Identificación:**
1. Verificar ACF/PACF en lags estacionales (m, 2m, 3m...)
2. Picos significativos en lags múltiplos de m → componente estacional
3. Diferenciación estacional si ACF decrece lento en lags estacionales

**Configuración común por frecuencia:**
- **Mensual**: SARIMA(p,d,q)(P,D,Q)₁₂
- **Trimestral**: SARIMA(p,d,q)(P,D,Q)₄
- **Semanal**: SARIMA(p,d,q)(P,D,Q)₅₂
- **Diaria con semana**: SARIMA(p,d,q)(P,D,Q)₇

**Ejemplo típico:** ARIMA(1,1,1)(1,1,1)₁₂
- AR(1) y MA(1) no estacional
- AR(1) y MA(1) estacional
- Una diferenciación regular y una estacional

### 5.3 ARIMAX / SARIMAX

**Variables Exógenas (X):**
- **Calendario**: días festivos, eventos especiales
- **Económicas**: precio, promociones, competencia
- **Climáticas**: temperatura, precipitación
- **Lags de otras series**: ventas de productos relacionados

**Consideraciones:**
1. Variables exógenas deben estar disponibles en horizonte de pronóstico
2. Si se desconocen valores futuros → pronosticar las X también
3. Evaluar causalidad: X debe preceder temporalmente a Y
4. Test de Granger causality

**Construcción:**
```
Y_t = β₀ + β₁X₁_t + ... + βₖXₖ_t + η_t
donde η_t ~ SARIMA(p,d,q)(P,D,Q)ₘ
```

### 5.4 ETS (Error, Trend, Seasonal)

**Componentes:**
- **Error**: Aditivo (A) o Multiplicativo (M)
- **Trend**: None (N), Aditivo (A), Aditivo Damped (Ad), Multiplicativo (M), Multiplicativo Damped (Md)
- **Seasonal**: None (N), Aditivo (A), Multiplicativo (M)

**30 posibles combinaciones**, las más comunes:

| Modelo | Notación | Uso |
|--------|----------|-----|
| Simple Exponential Smoothing | ETS(A,N,N) | Sin tendencia ni estacionalidad |
| Holt's Linear | ETS(A,A,N) | Tendencia lineal |
| Damped Trend | ETS(A,Ad,N) | Tendencia que se amortigua |
| Holt-Winters Aditivo | ETS(A,A,A) | Tendencia y estacionalidad aditiva |
| Holt-Winters Multiplicativo | ETS(A,A,M) | Estacionalidad multiplicativa |

**Selección automática:**
- Probar todas las combinaciones
- Seleccionar por AICc
- Verificar restricciones de admisibilidad (parámetros en rango válido)

**Ventajas:**
- Framework unificado de suavizamiento
- Intervalos de predicción directos
- Rápido de entrenar
- Interpretable

### 5.5 Prophet (Meta/Facebook)

**Componentes:**
```
y(t) = g(t) + s(t) + h(t) + εₜ
```
- **g(t)**: Tendencia (lineal o logística)
- **s(t)**: Estacionalidad (Fourier series)
- **h(t)**: Efectos de días festivos
- **εₜ**: Error

**Configuración:**

**Tendencia:**
- `growth='linear'`: Para series que crecen linealmente
- `growth='logistic'`: Para series con saturación (cap, floor)
- `changepoint_prior_scale`: Flexibilidad de cambios de tendencia (default 0.05)
- `changepoint_range`: Proporción de datos para detectar changepoints (default 0.8)

**Estacionalidad:**
- `yearly_seasonality`: Auto, True, False, o entero (orden Fourier)
- `weekly_seasonality`: idem
- `daily_seasonality`: idem
- `seasonality_mode`: 'additive' o 'multiplicative'
- `seasonality_prior_scale`: Flexibilidad estacional (default 10)

**Días festivos:**
```python
holidays = pd.DataFrame({
    'holiday': 'nombre',
    'ds': fecha,
    'lower_window': días antes,
    'upper_window': días después
})
```

**Variables adicionales:**
- `add_regressor()`: Para variables exógenas
- `add_seasonality()`: Estacionalidades custom

**Ventajas:**
- Maneja automáticamente missing data
- Robusto a outliers
- Excelente para series con estacionalidad compleja
- Incorpora conocimiento de dominio (festivos)

**Cuándo usar:**
- Series con múltiples estacionalidades
- Efectos de festivos importantes
- Cambios de tendencia
- Datos con gaps

### 5.6 TBATS y BATS

**TBATS**: Trigonometric, Box-Cox, ARMA errors, Trend, Seasonal

**Componentes:**
1. **Box-Cox**: Transformación de varianza
2. **Trend**: Con amortiguamiento
3. **ARMA**: Errores autocorrelacionados
4. **Seasonal**: Múltiples estacionalidades (Fourier)

**Modelo:**
```
y_t^(λ) = l_{t-1} + φb_{t-1} + Σ sⱼ,t + d_t
```

**Cuándo usar:**
- **Múltiples estacionalidades**: diaria + semanal, horaria + diaria + semanal
- Estacionalidad compleja
- Series con período largo (>200)
- Cuando SARIMA no captura patrones

**Ejemplo:** Electricidad con estacionalidad horaria (24) y semanal (168)

**Configuración:**
- `use.box.cox=TRUE`: Estimar λ
- `use.trend=TRUE`: Incluir tendencia
- `use.damped.trend=TRUE`: Tendencia amortiguada
- `seasonal.periods`: Vector de períodos estacionales
- `use.arma.errors=TRUE`: Modelar autocorrelación residual

**BATS vs TBATS:**
- BATS: Usa modelo de espacio de estados exponencial
- TBATS: Usa representación trigonométrica (más rápido para períodos grandes)

### 5.7 Dynamic Harmonic Regression

**Combinación de ARIMA con regresores de Fourier:**

```
Y_t = Σ[βₖsin(2πkt/m) + γₖcos(2πkt/m)] + η_t
donde η_t ~ ARIMA
```

**Ventajas sobre SARIMA:**
- No limitado por período estacional
- Más parsimonioso (menos parámetros)
- Maneja estacionalidad larga (m > 200)

**K (número de términos de Fourier):**
- K pequeño: suavizado, captura estacionalidad general
- K grande: más flexible, captura irregularidades
- K ≤ m/2
- Seleccionar por AICc

**Uso típico:**
- Series con múltiples estacionalidades
- Períodos estacionales largos
- Complemento a variables exógenas

---

## **FASE 6: MODELOS DE MACHINE LEARNING**

### 6.1 Feature Engineering para Series Temporales

**Features temporales básicos:**

**1. Lags**
```python
# Lags de la variable objetivo
lags = [1, 2, 3, 7, 14, 21, 30]  # Ejemplo
df['lag_1'] = df['y'].shift(1)
df['lag_7'] = df['y'].shift(7)
```

**Selección de lags:**
- Basado en ACF/PACF significativos
- Lags estacionales (m, 2m, 3m)
- Autorregresivos importantes del modelo ARIMA base
- Feature importance de modelo preliminar

**2. Rolling statistics (ventanas móviles)**
```python
windows = [7, 14, 30, 90]
for w in windows:
    df[f'rolling_mean_{w}'] = df['y'].rolling(w).mean()
    df[f'rolling_std_{w}'] = df['y'].rolling(w).std()
    df[f'rolling_min_{w}'] = df['y'].rolling(w).min()
    df[f'rolling_max_{w}'] = df['y'].rolling(w).max()
    df[f'rolling_median_{w}'] = df['y'].rolling(w).median()
    df[f'rolling_skew_{w}'] = df['y'].rolling(w).skew()
    df[f'rolling_kurt_{w}'] = df['y'].rolling(w).kurt()
```

**3. Expanding statistics (acumulativas)**
```python
df['expanding_mean'] = df['y'].expanding().mean()
df['expanding_std'] = df['y'].expanding().std()
```

**4. Diferencias y cambios**
```python
# Diferencias
df['diff_1'] = df['y'].diff(1)
df['diff_7'] = df['y'].diff(7)  # Diferencia estacional

# Porcentaje de cambio
df['pct_change_1'] = df['y'].pct_change(1)
df['pct_change_7'] = df['y'].pct_change(7)
```

**5. Features de calendario**
```python
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df['dayofyear'] = df.index.dayofyear
df['quarter'] = df.index.quarter
df['weekofyear'] = df.index.isocalendar().week
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
df['is_month_start'] = df.index.is_month_start.astype(int)
df['is_month_end'] = df.index.is_month_end.astype(int)
```

**6. Encoding cíclico (importante para ML)**
```python
# Para capturar naturaleza cíclica del tiempo
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
df['day_sin'] = np.sin(2 * np.pi * df.index.day / 31)
df['day_cos'] = np.cos(2 * np.pi * df.index.day / 31)
df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
```

**7. Features de festivos y eventos**
```python
# Días festivos
df['is_holiday'] = df.index.isin(holidays).astype(int)
df['days_to_holiday'] = # calcular distancia
df['days_since_holiday'] = # calcular distancia

# Eventos especiales
df['is_black_friday'] = ...
df['is_cyber_monday'] = ...
```

**8. Features de estacionalidad descompuesta**
```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['y'], model='additive', period=12)
df['seasonal_component'] = decomposition.seasonal
df['trend_component'] = decomposition.trend
```

**9. Interactions (interacciones)**
```python
# Ejemplo: interacción mes x año
df['month_year'] = df['month'] * df['year']

# Lag x variable exógena
df['lag1_x_promo'] = df['lag_1'] * df['promo']
```

**10. Domain-specific features**
- Eventos de negocio
- Promociones, campañas marketing
- Índices económicos
- Clima
- Precio de competidores

### 6.2 XGBoost / LightGBM / CatBoost

**Configuración para series temporales:**

**Validación temporal:**
```python
# TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_


]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

**XGBoost para series temporales:**
```python
import xgboost as xgb

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,  # Profundidad moderada
    'learning_rate': 0.01,  # Learning rate bajo
    'n_estimators': 1000,
    'subsample': 0.8,  # Prevenir overfitting
    'colsample_bytree': 0.8,
    'min_child_weight': 3,  # Regularización
    'gamma': 0,  # Complejidad mínima para split
    'reg_alpha': 0.1,  # L1 regularización
    'reg_lambda': 1,  # L2 regularización
    'early_stopping_rounds': 50,
    'eval_metric': 'rmse'
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)],
          verbose=False)
```

**LightGBM (más rápido):**
```python
import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,  # 2^max_depth - 1
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0.0,
    'early_stopping_rounds': 50
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          categorical_feature=categorical_features)
```

**CatBoost (mejor con categóricas):**
```python
from catboost import CatBoostRegressor

params = {
    'iterations': 1000,
    'learning_rate': 0.01,
    'depth': 6,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'early_stopping_rounds': 50,
    'use_best_model': True,
    'verbose': False
}

model = CatBoostRegressor(**params)
model.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          cat_features=categorical_features)
```

**Feature Importance:**
```python
# Analizar importancia
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# SHAP values para interpretabilidad
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Hyperparameter tuning:**
```python
from optuna import create_study

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

study = create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### 6.3 Random Forest

**Configuración:**
```python
from sklearn.ensemble import RandomForestRegressor

params = {
    'n_estimators': 500,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',  # o 0.33 para regresión
    'bootstrap': True,
    'oob_score': True,  # Out-of-bag score
    'n_jobs': -1,
    'random_state': 42
}

model = RandomForestRegressor(**params)
model.fit(X_train, y_train)
```

**Ventajas:**
- Menos propenso a overfitting que XGBoost
- No requiere mucha tuning
- Maneja bien outliers
- Feature importance confiable

**Desventajas:**
- Generalmente menos preciso que gradient boosting
- Memoria intensivo
- No extrapola fuera del rango de entrenamiento

### 6.4 LSTM / GRU (Deep Learning)

**Arquitectura básica LSTM:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Preparar datos en formato 3D: (samples, timesteps, features)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Lookback window
X_train, y_train = create_sequences(train_scaled, seq_length)
X_val, y_val = create_sequences(val_scaled, seq_length)

# Modelo
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

**Variantes LSTM:**

**1. Bidirectional LSTM:**
```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_length, n_features)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(1)
])
```

**2. Stacked LSTM:**
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
    LSTM(64, return_sequences=True),
    LSTM(32, return_sequences=False),
    Dense(1)
])
```

**3. Encoder-Decoder LSTM (Seq2Seq):**
```python
from tensorflow.keras.layers import RepeatVector, TimeDistributed

# Para pronósticos multi-step
model = Sequential([
    # Encoder
    LSTM(100, activation='relu', input_shape=(seq_length, n_features)),
    RepeatVector(forecast_horizon),
    # Decoder
    LSTM(100, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])
```

**GRU (más simple, a veces mejor):**
```python
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(128, return_sequences=True, input_shape=(seq_length, n_features)),
    Dropout(0.2),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
```

**Attention Mechanism:**
```python
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Uso
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
    AttentionLayer(),
    Dense(64, activation='relu'),
    Dense(1)
])
```

**Consideraciones importantes:**
1. **Normalización**: Siempre escalar datos (MinMaxScaler o StandardScaler)
2. **Secuencia length**: Típicamente 7-60 pasos
3. **Early stopping**: Esencial para evitar overfitting
4. **Hyperparámetros**: Units (32-256), layers (2-4), dropout (0.2-0.5)

### 6.5 N-BEATS (Neural Basis Expansion Analysis)

**Modelo state-of-the-art para forecasting univariado:**

```python
# Usando darts library
from darts.models import NBEATSModel
from darts import TimeSeries

# Convertir a TimeSeries
ts_train = TimeSeries.from_dataframe(train_df, time_col='date', value_cols='value')
ts_val = TimeSeries.from_dataframe(val_df, time_col='date', value_cols='value')

# Modelo
model = NBEATSModel(
    input_chunk_length=30,  # Lookback window
    output_chunk_length=7,  # Forecast horizon
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=32,
    generic_architecture=True,  # False para interpretable
    num_stacks=30,
    num_blocks=1,
    num_layers=4,
    layer_widths=256,
    expansion_coefficient_dim=5,
    trend_polynomial_degree=2
)

model.fit(ts_train, val_series=ts_val, verbose=True)
prediction = model.predict(n=7)
```

**Dos variantes:**
1. **Generic N-BEATS**: Completamente basado en datos
2. **Interpretable N-BEATS**: Descompone en trend y seasonality

**Ventajas:**
- SOTA en benchmarks M4, M5
- No requiere feature engineering
- Maneja múltiples estacionalidades
- Arquitectura específica para forecasting

### 6.6 Transformer / Temporal Fusion Transformer

**Temporal Fusion Transformer (TFT):**

```python
# Usando pytorch-forecasting
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl

# Preparar datos
max_encoder_length = 30
max_prediction_length = 7

training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="value",
    group_ids=["series_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["category"],
    static_reals=["static_feature"],
    time_varying_known_categoricals=["month", "dayofweek"],
    time_varying_known_reals=["time_idx", "holiday"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["value", "lag_7"],
    target_normalizer=GroupNormalizer(groups=["series_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
)

validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True)

# DataLoaders
train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

# Modelo
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,  # Quantiles
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4
)

# Entrenamiento
trainer = pl.Trainer(
    max_epochs=50,
    gpus=1,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback]
)

trainer.fit(tft, train_dataloader, val_dataloader)
```

**Ventajas TFT:**
- Variable importance integrada
- Intervalos de predicción (quantiles)
- Maneja variables estáticas, conocidas, y desconocidas
- Attention mechanism interpretable

**Informer / Autoformer:**
- Variantes de Transformer optimizadas para series largas
- Reducen complejidad O(n²) a O(n log n)

### 6.7 DeepAR (Autoregressive RNN)

**Para múltiples series temporales relacionadas:**

```python
# Usando GluonTS
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset

# Preparar datos
train_ds = ListDataset(
    [{"start": df.index[0], "target": df['value'].values} for df in train_dfs],
    freq="H"
)

# Modelo
estimator = DeepAREstimator(
    prediction_length=24,
    context_length=168,  # 7 días para datos horarios
    freq="H",
    num_layers=3,
    num_cells=40,
    cell_type='lstm',
    dropout_rate=0.1,
    use_feat_dynamic_real=True,
    use_feat_static_cat=True,
    cardinality=[10, 5],  # Cardinalidad de features categóricas
    embedding_dimension=[5, 3],
    trainer=Trainer(
        epochs=100,
        learning_rate=1e-3,
        batch_size=32,
        num_batches_per_epoch=50
    )
)

predictor = estimator.train(train_ds)
```

**Cuándo usar:**
- Múltiples series relacionadas
- Quieres aprovechar información cruzada
- Necesitas intervalos de predicción probabilísticos

---

## **FASE 7: VALIDACIÓN Y EVALUACIÓN**

### 7.1 Estrategias de Validación Temporal

**1. Train/Validation/Test Split**
```
|-------- Train --------|--Val--|--Test--|
                        ^       ^
                        |       |
                     Punto 1  Punto 2
```
- Train: 60-70%
- Validation: 15-20%
- Test: 15-20%

**2. Rolling Window (Fixed Size)**
```
Train: [1----30] -> Test: [31-37]
Train: [2----31] -> Test: [32-38]
Train: [3----32] -> Test: [33-39]
```

**3. Expanding Window**
```
Train: [1----30] -> Test: [31-37]
Train: [1----37] -> Test: [38-44]
Train: [1----44] -> Test: [45-51]
```

**4. Time Series Cross-Validation**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=30)

for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    # Entrenar y evaluar
```

**5. Blocked Cross-Validation**
```
|--Train--|--Gap--|--Test--| ... |--Train--|--Gap--|--Test--|
```
- Gap previene data leakage
- Útil cuando hay correlación temporal

### 7.2 Métricas de Evaluación

**Métricas de Error:**

**1. MAE (Mean Absolute Error)**
```
MAE = (1/n) Σ|y_i - ŷ_i|
```
- Interpretación: Error promedio en unidades originales
- Robusto a outliers
- No penaliza grandes errores proporcionalmente

**2. RMSE (Root Mean Squared Error)**
```
RMSE = √[(1/n) Σ(y_i - ŷ_i)²]
```
- Penaliza más los grandes errores
- En unidades originales
- Más sensible a outliers

**3. MAPE (Mean Absolute Percentage Error)**
```
MAPE = (100/n) Σ|{(y_i - ŷ_i) / y_i}|
```
- Porcentaje de error
- Fácil de interpretar
- **Problemas**: infinito cuando y=0, asimétrico (penaliza más over-forecast)

**4. SMAPE (Symmetric MAPE)**
```
SMAPE = (100/n) Σ|y_i - ŷ_i| / [(|y_i| + |ŷ_i|)/2]
```
- Soluciona asimetría de MAPE
- Rango: 0-200%
- Aún tiene problemas cerca de 0

**5. MASE (Mean Absolute Scaled Error)**
```
MASE = MAE / MAE_naive
donde MAE_naive es error de modelo ingenuo (seasonal naive)
```
- Independiente de escala
- MASE < 1: mejor que naive forecast
- Recomendado por Hyndman

**6. RMSSE (Root Mean Squared Scaled Error)**
```
RMSSE = √[MSE / MSE_naive]
```
- Similar a MASE pero con cuadrados
- Usado en competencia M5

**7. Quantile Loss (para predicciones probabilísticas)**
```
QL_τ = Σ[(y_i - q_τ)(τ - 1{y_i < q_τ})]
```
- τ = quantile (e.g., 0.1, 0.5, 0.9)
- Para intervalos de predicción

**Métricas de Información (modelos estadísticos):**

**8. AIC (Akaike Information Criterion)**
```
AIC = -2log(L) + 2k
```
- k = número de parámetros
- Menor es mejor
- Penaliza complejidad

**9. BIC (Bayesian Information Criterion)**
```
BIC = -2log(L) + k×log(T)
```
- Penaliza más la complejidad que AIC
- Prefiere modelos más simples

**10. AICc (AIC corregido)**
```
AICc = AIC + 2k(k+1)/(T-k-1)
```
- Para muestras pequeñas (T/k < 40)

**Métricas de Dirección:**

**11. DA (Directional Accuracy)**
```
DA = (1/n) Σ1{sign(Δy_i) = sign(Δŷ_i)}
```
- Porcentaje de veces que predice correctamente la dirección del cambio

**Selección de métrica según caso:**

| Caso | Métrica Recomendada | Razón |
|------|---------------------|-------|
| Series con ceros | MAE, MASE | MAPE indefinido |
| Comparar múltiples series | MASE, RMSSE | Independiente de escala |
| Interpretación del negocio | MAPE, MAE | Fácil de explicar |
| Optimización general | RMSE, MAE | Standard |
| Penalizar grandes errores | RMSE | Cuadrático |
| Predicción probabilística | Quantile Loss, CRPS | Evalúa intervalos |
| Trading/finanzas | DA, Sharpe Ratio | Dirección importa |

### 7.3 Diagnóstico de Residuos

**Tests esenciales para modelos estadísticos:**

**1. Test de Normalidad**

**Jarque-Bera:**
```
H0: Residuos son normales
Basado en skewness y kurtosis
```

**Shapiro-Wilk:**
```
H0: Residuos son normales
Más potente para muestras pequeñas (n < 50)
```

**Anderson-Darling:**
```
H0: Residuos siguen distribución especificada
Más sensible en las colas
```

**Visualización:**
- Q-Q plot
- Histograma con curva normal
- KDE plot

**2. Test de Autocorrelación en Residuos**

**Ljung-Box:**
```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(residuals, lags=20, return_df=True)
# p-value > 0.05 en todos los lags → No autocorrelación
```

**Durbin-Watson:**
```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
# DW ≈ 2: No autocorrelación
```

**ACF plot de residuos:**
- No debe haber picos significativos
- Todos dentro de bandas de confianza

**3. Test de Heterocedasticidad**

**ARCH Test:**
```python
from statsmodels.stats.diagnostic import het_arch

arch_test = het_arch(residuals)
# p-value > 0.05 → No heterocedasticidad (varianza constante)
```

**Breusch-Pagan:**
```python
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(residuals, exog)
# p-value > 0.05 → Homocedasticidad
```

**White Test:**
- Más general, no asume forma funcional

**Visualización:**
- Residuos vs fitted values
- Residuos vs tiempo
- Rolling std de residuos

**4. Resumen de Diagnóstico**

```python
def diagnostic_plots(residuals, lags=20):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuos vs tiempo
    axes[0,0].plot(residuals)
    axes[0,0].axhline(0, color='red', linestyle='--')
    axes[0,0].set_title('Residuals over Time')
    
    # 2. Histograma + normal
    axes[0,1].hist(residuals, bins=30, density=True, alpha=0.7)
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0,1].plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2)
    axes[0,1].set_title('Histogram + Normal Distribution')
    
    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot')
    
    # 4. ACF
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=lags, ax=axes[1,1])
    axes[1,1].set_title('ACF of Residuals')
    
    plt.tight_layout()
    plt.show()
```

**Criterios de un buen modelo:**
✓ Residuos con media ≈ 0
✓ Varianza constante en el tiempo
✓ No autocorrelación (ruido blanco)
✓ Aproximadamente normales (deseable pero no crítico)
✓ No patrones sistemáticos

---

## **FASE 8: ENSAMBLES Y STACKING**

### 8.1 Tipos de Ensambles

**1. Simple Average**
```python
forecast_avg = (forecast_model1 + forecast_model2 + forecast_model3) / 3
```
- Baseline simple
- Reduce varianza
- Funciona sorprendentemente bien

**2. Weighted Average**
```python
weights = [0.5, 0.3, 0.2]  # Basado en performance en validación
forecast_weighted = (w1*f1 + w2*f2 + w3*f3)
```

**Optimizar pesos:**
```python
from scipy.optimize import minimize

def objective(weights):
    forecast = sum(w * f for w, f in zip(weights, forecasts))
    return rmse(y_val, forecast)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(n_models)]

result = minimize(objective, x0=initial_weights, 
                  bounds=bounds, constraints=constraints)
optimal_weights = result.x
```

**3. Median**
```python
forecast_median = np.median([f1, f2, f3], axis=0)
```
- Más robusto a outliers que el promedio

**4. Trimmed Mean**
```python
from scipy.stats import trim_mean

forecast_trim = trim_mean([f1, f2, f3], proportiontocut=0.1, axis=0)
```
- Elimina extremos, combina robustez con eficiencia

### 8.2 Stacking

**Nivel 1: Modelos Base**
```python
# Entrenar modelos diversos
models = {
    'arima': auto_arima(...),
    'prophet': Prophet(...),
    'xgb': XGBRegressor(...),
    'lstm': LSTM_model(...)
}

# Generar predicciones out-of-fold
meta_features_train = []
meta_features_test = []

for name, model in models.items():
    # Cross-validation para generar meta-features
    oof_preds = cross_val_predict_timeseries(model, X_train, y_train)
    meta_features_train.append(oof_preds)
    
    # Predicciones en test
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    meta_features_test.append(test_preds)

# Convertir a arrays
X_meta_train = np.column_stack(meta_features_train)
X_meta_test = np.column_stack(meta_features_test)
```

**Nivel 2: Meta-Modelo**
```python
# Opciones para meta-modelo
meta_model = Ridge(alpha=1.0)  # Simple y funciona bien
# O
meta_model = XGBRegressor(max_depth=2, learning_rate=0.1)  # Más flexible

meta_model.fit(X_meta_train, y_train)
final_forecast = meta_model.predict(X_meta_test)
```

**Variación: Stacking con Features Originales**
```python
# Incluir features originales + predicciones
X_meta_train_extended = np.column_stack([X_train, X_meta_train])
X_meta_test_extended = np.column_stack([X_test, X_meta_test])

meta_model.fit(X_meta_train_extended, y_train)
```

### 8.3 Ensambles Específicos para Series Temporales

**1. Forecast Combination por Horizonte**
```python
# Diferentes modelos pueden ser mejores en diferentes horizontes
weights_by_horizon = {
    1: {'arima': 0.6, 'xgb': 0.4},     # h=1
    7: {'prophet': 0.5, 'lstm': 0.5},   # h=7
    30: {'tbats': 0.7, 'ets': 0.3}      # h=30
}
```

**2. Adaptive Combinations**
```python
# Pesos cambian basados en performance reciente
def adaptive_weights(errors_recent, decay=0.9):
    mse = [np.mean(e**2) for e in errors_recent]
    inv_mse = [1/m if m > 0 else 1e6 for m in mse]
    weights = inv_mse / np.sum(inv_mse)
    return weights
```

**3. Forecast Encompassing Test**
```python
# Test estadístico para determinar si un modelo abarca a otro
from statsmodels.regression.linear_model import OLS

# Si modelo 1 "encompass" modelo 2:
# y_true = α + β*forecast_1 + γ*forecast_2 + ε
# H0: γ = 0

model = OLS(y_true, np.column_stack([forecast_1, forecast_2]))
results = model.fit()
# Si γ no significativo, modelo 1 es suficiente
```

---

## **FASE 9: INTERVALOS DE PREDICCIÓN**

### 9.1 Métodos Paramétricos

**1. Para modelos ARIMA/SARIMA**
```python
# Intervalos automáticos
forecast_result = model.get_forecast(steps=horizon)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95% CI
```

**Fórmula:**
```
ŷ_t ± z_α/2 × σ̂ × √(1 + Σψ_j²)
donde ψ
_j son coeficientes MA infinitos
```

**2. Para modelos ETS**
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(train, seasonal_periods=12, 
                             trend='add', seasonal='add')
fit = model.fit()
forecast = fit.forecast(steps=horizon)
simulation = fit.simulate(nsimulations=horizon, repetitions=1000)

# Intervalos por percentiles de simulación
lower = np.percentile(simulation, 2.5, axis=0)
upper = np.percentile(simulation, 97.5, axis=0)
```

**3. Prophet**
```python
# Intervalos automáticos por simulación
future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

# Columnas: yhat, yhat_lower, yhat_upper
plt.plot(forecast['ds'], forecast['yhat'])
plt.fill_between(forecast['ds'], 
                 forecast['yhat_lower'], 
                 forecast['yhat_upper'], 
                 alpha=0.3)
```

**Ajustar ancho de intervalo:**
```python
model = Prophet(interval_width=0.95)  # 95% CI (default 80%)
```

### 9.2 Métodos No Paramétricos

**1. Bootstrap de Residuos**
```python
def bootstrap_forecast(model, X_train, y_train, X_test, n_simulations=1000):
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Obtener residuos
    train_pred = model.predict(X_train)
    residuals = y_train - train_pred
    
    # Simulaciones
    forecasts = []
    for _ in range(n_simulations):
        # Resample residuos con reemplazo
        bootstrap_residuals = np.random.choice(residuals, 
                                               size=len(X_test), 
                                               replace=True)
        
        # Predicción + residuo bootstrap
        base_forecast = model.predict(X_test)
        bootstrap_forecast = base_forecast + bootstrap_residuals
        forecasts.append(bootstrap_forecast)
    
    forecasts = np.array(forecasts)
    
    # Percentiles para intervalos
    lower = np.percentile(forecasts, 2.5, axis=0)
    median = np.percentile(forecasts, 50, axis=0)
    upper = np.percentile(forecasts, 97.5, axis=0)
    
    return median, lower, upper
```

**2. Block Bootstrap (para dependencia temporal)**
```python
from arch.bootstrap import CircularBlockBootstrap

def block_bootstrap_forecast(residuals, block_size=10, n_simulations=1000):
    bs = CircularBlockBootstrap(block_size, residuals)
    forecasts = []
    
    for data in bs.bootstrap(n_simulations):
        bootstrap_residuals = data[0][0][:forecast_horizon]
        forecast = base_forecast + bootstrap_residuals
        forecasts.append(forecast)
    
    return np.array(forecasts)
```

**3. Quantile Regression**
```python
from sklearn.ensemble import GradientBoostingRegressor

# Entrenar modelos para diferentes quantiles
quantiles = [0.025, 0.5, 0.975]
models = {}

for q in quantiles:
    models[q] = GradientBoostingRegressor(
        loss='quantile',
        alpha=q,
        n_estimators=100,
        max_depth=3
    )
    models[q].fit(X_train, y_train)

# Predicciones
lower = models[0.025].predict(X_test)
median = models[0.5].predict(X_test)
upper = models[0.975].predict(X_test)
```

**4. Quantile Regression con LightGBM**
```python
import lightgbm as lgb

# Quantile 0.025
lgb_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.025)
lgb_lower.fit(X_train, y_train)

# Quantile 0.5 (mediana)
lgb_median = lgb.LGBMRegressor(objective='quantile', alpha=0.5)
lgb_median.fit(X_train, y_train)

# Quantile 0.975
lgb_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.975)
lgb_upper.fit(X_train, y_train)
```

**5. Conformal Prediction**
```python
def conformal_prediction(model, X_calib, y_calib, X_test, alpha=0.1):
    """
    Intervalos de predicción conformes
    Garantía de cobertura sin supuestos distribucionales
    """
    # Predicciones en calibración
    calib_pred = model.predict(X_calib)
    
    # Scores de no conformidad (residuos absolutos)
    scores = np.abs(y_calib - calib_pred)
    
    # Quantile de scores
    q = np.quantile(scores, 1 - alpha)
    
    # Predicción en test
    test_pred = model.predict(X_test)
    
    # Intervalos
    lower = test_pred - q
    upper = test_pred + q
    
    return test_pred, lower, upper
```

### 9.3 Métodos para Deep Learning

**1. MC Dropout (Monte Carlo Dropout)**
```python
def predict_with_uncertainty(model, X, n_iter=100):
    """
    Mantener dropout activo durante predicción
    """
    predictions = []
    
    for _ in range(n_iter):
        # Dropout activo en inference
        pred = model(X, training=True)
        predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    lower = np.percentile(predictions, 2.5, axis=0)
    upper = np.percentile(predictions, 97.5, axis=0)
    
    return mean, lower, upper, std
```

**2. Ensemble de Redes**
```python
# Entrenar múltiples redes con diferentes inicializaciones
n_models = 10
models = []

for i in range(n_models):
    model = create_lstm_model()
    model.fit(X_train, y_train, epochs=100, verbose=0)
    models.append(model)

# Predicciones
predictions = np.array([model.predict(X_test) for model in models])

mean = predictions.mean(axis=0)
lower = np.percentile(predictions, 2.5, axis=0)
upper = np.percentile(predictions, 97.5, axis=0)
```

**3. Quantile Loss para LSTM**
```python
import tensorflow as tf

def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q*e, (q-1)*e))

# Entrenar tres modelos
model_lower = create_lstm_model()
model_lower.compile(optimizer='adam', 
                    loss=lambda y,f: quantile_loss(0.025, y, f))

model_median = create_lstm_model()
model_median.compile(optimizer='adam', 
                     loss=lambda y,f: quantile_loss(0.5, y, f))

model_upper = create_lstm_model()
model_upper.compile(optimizer='adam', 
                    loss=lambda y,f: quantile_loss(0.975, y, f))
```

### 9.4 Evaluación de Intervalos

**1. Coverage (Cobertura)**
```python
def coverage(y_true, lower, upper):
    """
    Proporción de valores reales dentro del intervalo
    Ideal: coverage ≈ nivel de confianza (e.g., 0.95)
    """
    in_interval = (y_true >= lower) & (y_true <= upper)
    return in_interval.mean()
```

**2. Interval Width**
```python
def mean_interval_width(lower, upper):
    """
    Ancho promedio del intervalo
    Más estrecho es mejor (dado coverage adecuado)
    """
    return (upper - lower).mean()
```

**3. Winkler Score**
```python
def winkler_score(y_true, lower, upper, alpha=0.05):
    """
    Combina width y coverage
    Menor es mejor
    """
    width = upper - lower
    penalty_lower = (2/alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2/alpha) * (y_true - upper) * (y_true > upper)
    
    return (width + penalty_lower + penalty_upper).mean()
```

**4. Calibration Plot**
```python
def calibration_plot(y_true, predictions_quantiles, quantiles):
    """
    Verificar calibración de quantiles predichos
    """
    observed_freq = []
    
    for q in quantiles:
        pred_q = predictions_quantiles[q]
        freq = (y_true <= pred_q).mean()
        observed_freq.append(freq)
    
    plt.figure(figsize=(8, 6))
    plt.plot(quantiles, quantiles, 'k--', label='Perfect calibration')
    plt.plot(quantiles, observed_freq, 'ro-', label='Observed')
    plt.xlabel('Predicted Quantile')
    plt.ylabel('Observed Frequency')
    plt.legend()
    plt.title('Calibration Plot')
```

---

## **FASE 10: CASOS ESPECIALES**

### 10.1 Series con Múltiples Estacionalidades

**Ejemplo: Electricidad (diaria + semanal)**

**Método 1: TBATS**
```python
from tbats import TBATS

estimator = TBATS(
    seasonal_periods=[24, 168],  # 24h diaria, 168h semanal
    use_box_cox=True,
    use_trend=True,
    use_damped_trend=True,
    use_arma_errors=True
)

model = estimator.fit(train)
forecast = model.forecast(steps=168)  # 1 semana
```

**Método 2: Dynamic Harmonic Regression**
```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf

def fourier_terms(t, period, K):
    """Generar términos de Fourier"""
    terms = np.column_stack([
        np.sin(2 * np.pi * k * t / period) for k in range(1, K+1)
    ] + [
        np.cos(2 * np.pi * k * t / period) for k in range(1, K+1)
    ])
    return terms

# Para múltiples estacionalidades
t = np.arange(len(train))
daily_terms = fourier_terms(t, period=24, K=3)
weekly_terms = fourier_terms(t, period=168, K=2)

X_train = np.column_stack([daily_terms, weekly_terms])

# ARIMAX con términos de Fourier
model = ARIMA(train, exog=X_train, order=(1,0,1))
fit = model.fit()

# Forecast
t_test = np.arange(len(train), len(train) + horizon)
daily_test = fourier_terms(t_test, period=24, K=3)
weekly_test = fourier_terms(t_test, period=168, K=2)
X_test = np.column_stack([daily_test, weekly_test])

forecast = fit.forecast(steps=horizon, exog=X_test)
```

**Método 3: Prophet**
```python
# Prophet maneja múltiples estacionalidades automáticamente
model = Prophet()
model.add_seasonality(name='daily', period=1, fourier_order=5)
model.add_seasonality(name='weekly', period=7, fourier_order=3)
model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

model.fit(df)
```

### 10.2 Series Intermitentes / Con Muchos Ceros

**Características:**
- Demanda esporádica (retail, spare parts)
- Muchos períodos con valor 0
- Métodos tradicionales fallan

**Método 1: Croston's Method**
```python
def croston_forecast(y, h):
    """
    Croston's method para demanda intermitente
    """
    # Identificar demandas no-cero
    demand = y[y > 0]
    intervals = np.diff(np.where(y > 0)[0])
    
    if len(demand) == 0:
        return np.zeros(h)
    
    # SES en demanda y intervalos
    alpha = 0.1
    
    # Nivel de demanda
    z = demand[0]
    for d in demand[1:]:
        z = alpha * d + (1 - alpha) * z
    
    # Intervalo entre demandas
    x = intervals[0] if len(intervals) > 0 else 1
    for i in intervals[1:]:
        x = alpha * i + (1 - alpha) * x
    
    # Forecast
    forecast = z / x
    return np.repeat(forecast, h)
```

**Método 2: TSB (Teunter-Syntetos-Babai)**
```python
def tsb_forecast(y, alpha_demand=0.1, alpha_prob=0.1):
    """
    TSB method - mejora sobre Croston
    """
    z = y[0]  # Nivel de demanda
    p = 1 if y[0] > 0 else 0  # Probabilidad de demanda
    
    forecasts = []
    
    for t in range(1, len(y)):
        # Forecast
        forecast = p * z
        forecasts.append(forecast)
        
        # Update
        if y[t] > 0:
            z = alpha_demand * y[t] + (1 - alpha_demand) * z
            p = alpha_prob * 1 + (1 - alpha_prob) * p
        else:
            p = alpha_prob * 0 + (1 - alpha_prob) * p
    
    return np.array(forecasts)
```

**Método 3: Two-Stage Modeling**
```python
# Etapa 1: Predecir probabilidad de demanda (clasificación)
from sklearn.ensemble import RandomForestClassifier

y_binary = (y > 0).astype(int)
clf = RandomForestClassifier()
clf.fit(X_train, y_binary_train)
prob_demand = clf.predict_proba(X_test)[:, 1]

# Etapa 2: Predecir cantidad dado que hay demanda (regresión)
from sklearn.ensemble import RandomForestRegressor

# Solo entrenar con observaciones no-cero
mask = y_train > 0
X_train_nonzero = X_train[mask]
y_train_nonzero = y_train[mask]

reg = RandomForestRegressor()
reg.fit(X_train_nonzero, y_train_nonzero)
quantity_forecast = reg.predict(X_test)

# Forecast final
final_forecast = prob_demand * quantity_forecast
```

**Método 4: Zero-Inflated Models**
```python
# Usando statsmodels
from statsmodels.discrete.count_model import ZeroInflatedPoisson

# Para datos de conteo con exceso de ceros
model = ZeroInflatedPoisson(y_train, X_train, exog_infl=X_train)
fit = model.fit()
forecast = fit.predict(X_test)
```

### 10.3 Series con Cambios Estructurales / Breakpoints

**Detección de Breakpoints:**

**Método 1: PELT (Pruned Exact Linear Time)**
```python
import ruptures as rpt

# Detectar cambios de nivel
algo = rpt.Pelt(model="rbf").fit(signal)
breakpoints = algo.predict(pen=10)

# Visualizar
rpt.display(signal, breakpoints)
```

**Método 2: Binary Segmentation**
```python
algo = rpt.Binseg(model="l2").fit(signal)
breakpoints = algo.predict(n_bkps=3)  # 3 breakpoints
```

**Método 3: Zivot-Andrews Test**
```python
from arch.unitroot import ZivotAndrews

za = ZivotAndrews(y)
print(za.summary())
# Detecta breakpoint endógeno + test de raíz unitaria
```

**Modelado con Breakpoints:**

**1. Segmented Regression**
```python
import pwlf

# Piecewise linear fit
my_pwlf = pwlf.PiecewiseLinFit(x, y)
breaks = my_pwlf.fit(n_segments=3)

# Forecast por segmento
x_test = np.array([...])
y_forecast = my_pwlf.predict(x_test)
```

**2. Intervention Analysis (ARIMAX con variables dummy)**
```python
# Crear dummy para punto de quiebre
dummy_break = np.zeros(len(y))
dummy_break[breakpoint_index:] = 1  # Step function

# Level shift
dummy_level = dummy_break

# Slope change
dummy_slope = np.arange(len(y)) * dummy_break

# ARIMAX con intervención
exog = np.column_stack([dummy_level, dummy_slope, other_vars])
model = ARIMA(y, exog=exog, order=(1,1,1))
fit = model.fit()
```

**3. Cambio de Régimen (Regime Switching)**
```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Markov Switching
mod = MarkovRegression(
    y, 
    k_regimes=2,  # 2 regímenes
    switching_variance=True,
    order=1  # AR(1)
)
res = mod.fit()

# Probabilidad de estar en cada régimen
regime_probs = res.smoothed_marginal_probabilities
```

### 10.4 Series con Valores Extremos / Outliers

**Tratamiento robusto:**

**1. Robust ARIMA**
```python
from statsmodels.robust import mad

def robust_arima(y, order):
    """
    ARIMA robusto a outliers usando M-estimators
    """
    model = ARIMA(y, order=order)
    fit = model.fit(method='bfgs', 
                    optim_score='approx',
                    optim_hessian='approx')
    
    # Identificar outliers por MAD
    residuals = fit.resid
    median_resid = np.median(residuals)
    mad_resid = mad(residuals)
    threshold = 3 * mad_resid
    
    outliers = np.abs(residuals - median_resid) > threshold
    
    # Re-estimar sin outliers
    y_clean = y.copy()
    y_clean[outliers] = np.nan
    y_clean = pd.Series(y_clean).interpolate()
    
    model_clean = ARIMA(y_clean, order=order)
    fit_clean = model_clean.fit()
    
    return fit_clean
```

**2. Prophet con outlier detection**
```python
model = Prophet(
    interval_width=0.95,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(df)

# Detectar outliers
forecast = model.predict(df)
residuals = df['y'] - forecast['yhat']
threshold = 3 * residuals.std()
outliers = np.abs(residuals) > threshold

# Remover y re-entrenar
df_clean = df[~outliers]
model.fit(df_clean)
```

**3. Robust Forecasting con Huber Loss**
```python
from sklearn.linear_model import HuberRegressor

# Huber loss es robusto a outliers
model = HuberRegressor(epsilon=1.35, max_iter=100)
model.fit(X_train, y_train)
```

### 10.5 Series Jerárquicas / Múltiples Series Relacionadas

**Ejemplo: Ventas Total → Por Región → Por Tienda**

**Método 1: Bottom-Up**
```python
# Pronosticar nivel más bajo, agregar hacia arriba
forecasts_stores = {}
for store in stores:
    model = auto_arima(data[store])
    forecasts_stores[store] = model.predict(h=horizon)

# Agregación
forecast_region = sum(forecasts_stores[s] for s in stores_region)
forecast_total = sum(forecasts_stores.values())
```

**Método 2: Top-Down con Proporciones**
```python
# Pronosticar total
model_total = auto_arima(data_total)
forecast_total = model.predict(h=horizon)

# Distribuir por proporciones históricas
proportions = data_stores.mean(axis=0) / data_total.mean()
forecasts_stores = forecast_total * proportions
```

**Método 3: Middle-Out**
```python
# Pronosticar nivel intermedio (regiones)
# Desagregar hacia abajo, agregar hacia arriba
```

**Método 4: Optimal Reconciliation (MinT)**
```python
# Usando hts package o scikit-hts
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.transformations.hierarchical.reconcile import Reconciler

# Definir jerarquía
hierarchy = {
    'total': ['region_A', 'region_B'],
    'region_A': ['store_1', 'store_2'],
    'region_B': ['store_3', 'store_4']
}

# Base forecasts
base_forecasts = forecast_all_series(data, hierarchy)

# Reconciliar usando MinTrace
reconciler = Reconciler(method="mint_shrink")
reconciled = reconciler.fit_transform(base_forecasts)
```

**Ventaja:** Reconciliación asegura coherencia (suma correcta) y mejora precisión

### 10.6 Series de Alta Frecuencia (Intraday)

**Desafíos:**
- Múltiples estacionalidades (hora, día, semana)
- Microstructure noise
- Volatilidad intraday patterns

**Consideraciones:**

**1. Patrones Intraday**
```python
# U-shape en volumen (alta actividad inicio/fin del día)
df['hour'] = df.index.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Dummy por hora
df['is_market_open'] = df['hour'].between(9, 16).astype(int)
df['is_lunch'] = df['hour'].between(12, 13).astype(int)
```

**2. Volatility Modeling (GARCH para varianza)**
```python
from arch import arch_model

# Modelar media
returns = df['price'].pct_change().dropna()

# GARCH(1,1) para volatilidad
model = arch_model(returns, vol='Garch', p=1, q=1)
fit = model.fit()

# Forecast de volatilidad
vol_forecast = fit.forecast(horizon=10)
```

**3. Realized Volatility**
```python
# Agregación de datos de alta frecuencia
def realized_volatility(returns_5min):
    """
    RV = suma de retornos cuadrados intraday
    """
    return np.sqrt(np.sum(returns_5min**2))

# Por día
daily_rv = returns_5min.groupby(returns_5min.index.date).apply(realized_volatility)
```

---

## **FASE 11: PRODUCCIÓN Y MONITOREO**

### 11.1 Pipeline de Producción

**Código de Pipeline:**

```python
import logging
from datetime import datetime
import joblib

class ForecastPipeline:
    def __init__(self, model_path, config):
        self.model = joblib.load(model_path)
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logging.basicConfig(
            filename=f'forecast_{datetime.now():%Y%m%d}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def validate_data(self, df):
        """Validar calidad de datos de entrada"""
        checks = {
            'missing_values': df.isnull().sum().sum() == 0,
            'date_continuity': self._check_date_continuity(df),
            'value_range': df['value'].between(self.config['min_val'], 
                                               self.config['max_val']).all(),
            'no_duplicates': not df.index.duplicated().any()
        }
        
        for check, result in checks.items():
            if not result:
                self.logger.error(f"Data validation failed: {check}")
                return False
        
        return True
    
    def _check_date_continuity(self, df):
        """Verificar que no hay gaps en fechas"""
        expected_freq = pd.infer_freq(df.index)
        full_range = pd.date_range(df.index[0], df.index[-1], freq=expected_freq)
        return len(full_range) == len(df)
    
    def generate_forecast(self, df, horizon=30):
        """Generar pronóstico"""
        try:
            # Feature engineering
            features = self.create_features(df)
            
            # Predict
            forecast = self.model.predict(features)
            
            # Validate forecast
            if not self.validate_forecast(forecast):
                self.logger.warning("Forecast validation failed, using fallback")
                forecast = self.fallback_forecast(df, horizon)
            
            self.logger.info(f"Forecast generated successfully for horizon={horizon}")
            return forecast
            
        except Exception as e:
            self.logger.error(f"Forecast generation failed: {str(e)}")
            return self.fallback_forecast(df, horizon)
    
    def validate_forecast(self, forecast):
        """Validar que el pronóstico es razonable"""
        checks = {
            'no_nans': not np.isnan(forecast).any(),
            'positive': (forecast >= 0).all(),
            'reasonable_range': (forecast <= self.config['max_forecast']).all(),
            'no_extreme_jumps': self._check_jumps(forecast)
        }
        return all(checks.values())
    
    def _check_jumps(self, forecast):
        """Verificar que no hay cambios excesivos entre períodos"""
        pct_change = np.abs(np.diff(forecast) / forecast[:-1])
        return (pct_change < self.config['max_pct_change']).all()
    
    def fallback_forecast(self, df, horizon):
        """Forecast de respaldo (naive estacional)"""
        seasonal_period = self.config['seasonal_period']
        return df['value'].iloc[-seasonal_period:].values[:horizon]
    
    def monitor_performance(self, forecast, actual):
        """Monitorear performance del modelo"""
        mae = mean_absolute_error(actual, forecast)
        mape = mean_absolute_percentage_error(actual, forecast)
        
        # Log métricas
        self.logger.info(f"MAE: {mae:.2f}, MAPE: {mape:.2%}")
        
        # Alertas si performance degrada
        if mae > self.config['mae_threshold']:
            self.logger.warning(f"MAE exceeded threshold: {mae} > {self.config['mae_threshold']}")
            self.trigger_retrain()
        
        # Guardar métricas para tracking
        self.save_metrics({
            'timestamp': datetime.now(),
            'mae': mae,
            'mape': mape,
            'forecast_horizon': len(forecast)
        })
    
    def trigger_retrain(self):
        """Iniciar proceso de re-entrenamiento"""
        self.logger.info("Triggering model retraining")
        # Llamar a script de reentrenamiento o queue de tareas
        # subprocess.run(['python', 'retrain.py'])
```

### 11.2 Monitoreo de Drift

**Concept Drift:** Cuando la relación entre features y target cambia

**1. Monitorear Distribución de Features**
```python
from scipy.stats import ks_2samp

def detect_feature_drift(train_features, prod_features, threshold=0.05):
    """
    KS test para detectar cambio en distribución
    """
    drift_detected = {}
    
    for col in train_features.columns:
        statistic, p_value = ks_2samp(train_features[col], prod_features[col])
        drift_detected[col] = p_value < threshold
    
    return drift_detected
```

**2. Monitorear Performance del Modelo**