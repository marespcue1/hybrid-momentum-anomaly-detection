# Sistema híbrido de detección de anomalías y Momentum Trading
# Hybrid Anomaly Detection & Momentum Trading System

---

## Versión en español

### Descripción

Este repositorio contiene el código fuente y los resultados experimentales del Trabajo Fin de Máster titulado:

**"Sistema híbrido de detección de anomalías y Momentum Trading con algoritmos de clustering (K-Means/DBSCAN)"**

El proyecto propone una arquitectura algorítmica híbrida que combina dos técnicas de aprendizaje no supervisado para mejorar la robustez de las estrategias de inversión basadas en el factor momentum:

1. **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) como filtro topológico para detectar y eliminar anomalías y ruido estructural del mercado.
2. **K-Means** como algoritmo de segmentación para identificar regímenes de mercado estables sobre el espacio de datos depurado.

### Estructura del proyecto

```
.
├── data/
│   ├── raw/                 # Datos brutos descargados
│   └── processed/           # Datos procesados y features construidas
├── src/
│   ├── config.py            # Configuración global (tickers, fechas, parámetros)
│   ├── data_loader.py       # Descarga y preprocesamiento de datos
│   ├── feature_engineering.py # Construcción de indicadores técnicos
│   ├── dbscan_filter.py     # Implementación de DBSCAN adaptativo
│   ├── kmeans_clustering.py # Segmentación con K-Means
│   ├── backtesting.py       # Simulación de la estrategia de inversión
│   └── visualization.py     # Funciones para generación de gráficos
├── notebooks/
│   ├── 01_obtencion_datos.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_ingenieria_caracteristicas_DBSCAN.ipynb
│   ├── 04_k-means_clustering.ipynb
│   ├── 05_walkforward_analysis.ipynb
|   └── 06_estudio_parametros.ipynb
├── figures/                 # Gráficos y visualizaciones generadas
├── results/                 # Resultados de walk-forward y grid search
├── docs/                    # Memoria del proyecto (PDF)
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Este archivo
```

### Requisitos

#### Python

El proyecto está desarrollado en Python 3.12. Las dependencias principales son:

```txt
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
ta>=0.9.0
```

Instala todas las dependencias con:

```bash
pip install -r requirements.txt
```

#### Datos

El sistema utiliza datos históricos de los siguientes activos financieros:

- **España**: SAN.MC, FER.MC, ACS.MC, ITX.MC, REP.MC, AMS.MC
- **EE.UU.**: AAPL, MSFT, NVDA, JPM, BLK, TSLA, DPZ, CAT, XOM
- **ETFs**: XLK, SMH, XLV, XLI, XLY, XLP, GLD, SPY, QQQ, EWP, MTUM, XLE, XLF, TLT, XLU

Los datos se descargan automáticamente desde Yahoo Finance mediante la librería `yfinance`.

### Metodología

#### 1. Adquisición y preprocesamiento

- Descarga de datos OHLCV (Open, High, Low, Close, Volume)
- Tratamiento de valores faltantes y atípicos
- Construcción de features: retornos, volatilidad, RSI, MACD, ATR

#### 2. Estandarización

- Aplicación de Rolling Z-Score con ventana de 252 días
- Homogeneización del espacio de características para evitar sesgos de escala

#### 3. Filtrado de anomalías (DBSCAN)

- Aplicación cross-sectional por fecha
- Calibración adaptativa de ε mediante k-distance
- Suelo topológico dinámico: `FLOOR = 0.2 * sqrt(d)`
- Identificación y purga de observaciones anómalas (label = -1)

#### 4. Segmentación de regímenes (K-Means)

- Aplicación sobre el espacio de datos depurado
- Número de clústeres: K = 5
- Inicialización con K-Means++
- Identificación de "acciones optimistas" (clúster de alto rendimiento)

#### 5. Backtesting y evaluación

- Estrategia de momentum: comprar activos del clúster optimista
- Rebalanceo periódico
- Métricas: Sharpe Ratio, CAGR, Maximum Drawdown
- Validación mediante walk-forward (in-sample / out-of-sample)

### Ejecución rápida

#### 1. Configuración

Edita `src/config.py` para ajustar:

```python
FECHA_INICIO_ANALISIS = "2017-01-01"
FECHA_FIN_ANALISIS = "2026-05-11"
TICKERS_TFM = {...}  # Activos a analizar
```

#### 2. Descarga y preprocesamiento

```python
from src.data_loader import descargar_datos
from src.feature_engineering import construir_features

# Descarga
close, high, low, volume = descargar_datos(ALL_TICKERS)

# Preprocesamiento y features
df = construir_features(close, high, low, volume)
```

#### 3. Filtrado DBSCAN

```python
from src.dbscan_filter import aplicar_dbscan_cross_sectional

df_dbscan = aplicar_dbscan_cross_sectional(df_clean)
```

#### 4. Segmentación K-Means

```python
from src.kmeans_clustering import aplicar_kmeans_purged

df_clusters = aplicar_kmeans_purged(df_dbscan)
```

#### 5. Backtesting

```python
from src.backtesting import run_walk_forward

resultados = run_walk_forward(df_clusters)
```

### Resultados principales

Los experimentos realizados demuestran que la arquitectura híbrida:

- Mejora el Sharpe Ratio respecto al modelo base (K-Means aislado)
- Reduce el Maximum Drawdown en escenarios de alta volatilidad
- Aumenta la estabilidad de los centroides de K-Means
- Elimina efectivamente el ruido estructural del mercado

### Visualizaciones

El sistema genera automáticamente las siguientes figuras:

- `epsilon_evolution.png` - Evolución temporal del parámetro ε de DBSCAN
- `k_distance_*.png` - Curvas de k-distance para calibración
- `clusters_*.png` - Distribución de clústeres en el espacio PCA
- `equity_curve.png` - Curva de capital de la estrategia
- `sharpe_scatter.png` - Distribución de Sharpe Ratio en walk-forward
- `degradation_bar.png` - Degradación de Sharpe entre train y test

### Autores

- **Marcel Espejo Cuenca**
- **Raúl Gómez Hernández**
- **Diego Martínez Díez**
  
---
**Institución**: Universidad Internacional de La Rioja
**Programa**: Máster Universitario en Ingeniería Matemática y Computación

---

## English Version

### Description

This repository contains the source code and experimental results of the Master's Thesis titled:

**"Hybrid anomaly detection and Momentum Trading system with clustering algorithms (K-Means/DBSCAN)"**

The project proposes a hybrid algorithmic architecture that combines two unsupervised learning techniques to improve the robustness of momentum-based investment strategies:

1. **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) as a topological filter to detect and remove market anomalies and structural noise.
2. **K-Means** as a segmentation algorithm to identify stable market regimes on the cleaned data space.

### Project Structure

```
.
├── data/
│   ├── raw/                 # Raw downloaded data
│   └── processed/           # Processed data and engineered features
├── src/
│   ├── config.py            # Global configuration (tickers, dates, parameters)
│   ├── data_loader.py       # Data download and preprocessing
│   ├── feature_engineering.py # Technical indicator construction
│   ├── dbscan_filter.py     # Adaptive DBSCAN implementation
│   ├── kmeans_clustering.py # K-Means segmentation
│   ├── backtesting.py       # Investment strategy simulation
│   └── visualization.py     # Plot generation functions
├── notebooks/
│   ├── 01_obtencion_datos.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_ingenieria_caracteristicas_DBSCAN.ipynb
│   ├── 04_k-means_clustering.ipynb
│   ├── 05_walkforward_analysis.ipynb
|   └── 06_estudio_parametros.ipynb
├── figures/                 # Generated plots and visualizations
├── results/                 # Walk-forward and grid search results
├── docs/                    # Thesis document (PDF)
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

### Requirements

#### Python

The project is developed in Python 3.12. Main dependencies are:

```txt
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
ta>=0.9.0
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

#### Data

The system uses historical data from the following financial assets:

- **Spain**: SAN.MC, FER.MC, ACS.MC, ITX.MC, REP.MC, AMS.MC
- **USA**: AAPL, MSFT, NVDA, JPM, BLK, TSLA, DPZ, CAT, XOM
- **ETFs**: XLK, SMH, XLV, XLI, XLY, XLP, GLD, SPY, QQQ, EWP, MTUM, XLE, XLF, TLT, XLU

Data is automatically downloaded from Yahoo Finance using the `yfinance` library.

### Methodology

#### 1. Data Acquisition and Preprocessing

- OHLCV data download (Open, High, Low, Close, Volume)
- Handling of missing values and outliers
- Feature engineering: returns, volatility, RSI, MACD, ATR

#### 2. Standardization

- Rolling Z-Score application with 252-day window
- Feature space homogenization to avoid scale bias

#### 3. Anomaly Filtering (DBSCAN)

- Cross-sectional application by date
- Adaptive ε calibration using k-distance
- Dynamic topological floor: `FLOOR = 0.2 * sqrt(d)`
- Identification and purging of anomalous observations (label = -1)

#### 4. Regime Segmentation (K-Means)

- Application on the cleaned data space
- Number of clusters: K = 5
- K-Means++ initialization
- Identification of "bullish stocks" (high-performance cluster)

#### 5. Backtesting and Evaluation

- Momentum strategy: buy assets from the bullish cluster
- Periodic rebalancing
- Metrics: Sharpe Ratio, CAGR, Maximum Drawdown
- Walk-forward validation (in-sample / out-of-sample)

### Quick Execution

#### 1. Configuration

Edit `src/config.py` to adjust:

```python
FECHA_INICIO_ANALISIS = "2017-01-01"
FECHA_FIN_ANALISIS = "2026-05-11"
TICKERS_TFM = {...}  # Assets to analyze
```

#### 2. Data Download and Preprocessing

```python
from src.data_loader import descargar_datos
from src.feature_engineering import construir_features

# Download
close, high, low, volume = descargar_datos(ALL_TICKERS)

# Preprocessing and features
df = construir_features(close, high, low, volume)
```

#### 3. DBSCAN Filtering

```python
from src.dbscan_filter import aplicar_dbscan_cross_sectional

df_dbscan = aplicar_dbscan_cross_sectional(df_clean)
```

#### 4. K-Means Segmentation

```python
from src.kmeans_clustering import aplicar_kmeans_purged

df_clusters = aplicar_kmeans_purged(df_dbscan)
```

#### 5. Backtesting

```python
from src.backtesting import run_walk_forward

resultados = run_walk_forward(df_clusters)
```

### Main Results

Experiments demonstrate that the hybrid architecture:

- Improves Sharpe Ratio compared to the base model (isolated K-Means)
- Reduces Maximum Drawdown in high volatility scenarios
- Increases K-Means centroid stability
- Effectively removes structural market noise

### Visualizations

The system automatically generates the following figures:

- `epsilon_evolution.png` - DBSCAN ε parameter temporal evolution
- `k_distance_*.png` - k-distance curves for calibration
- `clusters_*.png` - Cluster distribution in PCA space
- `equity_curve.png` - Strategy equity curve
- `sharpe_scatter.png` - Sharpe Ratio distribution in walk-forward
- `degradation_bar.png` - Sharpe degradation between train and test

### Authors

- **Marcel Espejo Cuenca** - Hybrid architecture design and implementation
- **Raúl Gómez Hernández** - Data acquisition and preparation
- **Diego Martínez Díez** - Exploratory analysis and state of the art

---
**Institution**: Universidad Internacional de La Rioja
**Program**: Master's Degree in Mathematical and Computational Engineering
