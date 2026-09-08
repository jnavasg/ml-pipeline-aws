# ML Pipeline — AWS

Pipeline de Machine Learning modular y reproducible, diseñado para desplegarse en AWS.

---

## Estructura del proyecto

```
.
├── data/
│   ├── raw/           # Datos originales sin modificar (no versionados)
│   └── processed/     # Features listos para entrenar (no versionados)
│
├── src/
│   ├── ingestion/     # Carga datos desde S3, DBs o APIs
│   │   └── ingestion.py
│   ├── preprocessing/ # Limpieza, encoding, escalado y split
│   │   └── preprocessing.py
│   ├── training/      # Entrenamiento y serialización del modelo
│   │   └── train.py
│   └── evaluation/    # Métricas, thresholds y reporte de calidad
│       └── evaluate.py
│
├── models/            # Modelos entrenados y métricas (no versionados)
├── notebooks/         # Exploración y prototipado
├── tests/             # Tests unitarios con pytest
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── .github/
│   └── workflows/
│       └── pipeline.yml   # CI: corre pytest en cada push
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Flujo del pipeline

```
[Fuente de datos]
      │
      ▼
 ingestion.py       ← descarga desde S3 / CSV / API → data/raw/
      │
      ▼
preprocessing.py    ← limpia, encoda, escala, hace split → data/processed/
      │
      ▼
   train.py         ← entrena el modelo → models/<nombre>.pkl
      │
      ▼
 evaluate.py        ← métricas + threshold check → models/metrics.json
```

---

## Configuración

### 1. Entorno virtual e instalación

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Credenciales AWS

Configura las credenciales antes de ejecutar pasos que accedan a S3:

```bash
# Opción A — variables de entorno
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Opción B — perfil local
aws configure --profile ml-pipeline
```

> En EC2 / ECS / Lambda usa un **IAM Role** en lugar de credenciales estáticas.

### 3. Variables de entorno del proyecto

Copia `.env.example` a `.env` y rellena los valores:

```bash
cp .env.example .env
```

---

## Ejecución manual de cada etapa

```bash
# Ingestion
python -m src.ingestion.ingestion

# Preprocessing
python -m src.preprocessing.preprocessing

# Training
python -m src.training.train

# Evaluation
python -m src.evaluation.evaluate
```

---

## Tests

```bash
pytest tests/ -v
```

El workflow de GitHub Actions los ejecuta automáticamente en cada push.

---

## Despliegue en AWS (pasos siguientes)

| Opción                  | Cuándo usarla                                      |
|-------------------------|----------------------------------------------------|
| **AWS Lambda**          | Inferencia bajo demanda, dataset pequeño           |
| **SageMaker Endpoints** | Inferencia en tiempo real con autoscaling          |
| **SageMaker Pipelines** | Orquestar todo el pipeline como DAG gestionado     |
| **Step Functions + S3** | Pipeline batch con control de estado               |
| **EC2 / ECS**           | Control total del entorno de entrenamiento         |

---

## Próximos pasos sugeridos

- [ ] Implementar la lógica real de cada etapa reemplazando los placeholders.
- [ ] Añadir `config.yaml` para externalizar hiperparámetros y rutas.
- [ ] Integrar MLflow o SageMaker Experiments para trazabilidad.
- [ ] Añadir serialización del `StandardScaler` para usarlo en inferencia.
- [ ] Configurar un endpoint de inferencia (Lambda o SageMaker).
- [ ] Añadir linting (`ruff`) y type-checking (`mypy`) al CI.
