## LATAM challenge

### EDA y calidad de datos
- Se detecto inconsistencia de tipos en `Vlo-I` y `Vlo-O`:
  - `Vlo-I`: `str` (65536) vs `int` (2670)
  - `Vlo-O`: `str` (65535) vs `float` (2671)
- **Impacto:** no afecta el proyecto, porque el pipeline/modelo usa columnas de fecha (p.ej. `Fecha-I` y `Fecha-O`), no `Vlo-I/Vlo-O`.

### Fix de `period_day`
- El codigo original no incluía límites horarios (ejemplo, no consideraba `05:00`).
- Resultado: **1230 filas** quedaron como `None`.
- Se corrigio incluyendo cotas (`>=` / `<=`) dentro de los rangos para evitar `None`.

### Modelo elegido
- Se eligio **Logistic Regression con balanceo**:
    - Recall ~**69%** para la clase `delay`, similar a **XGBoost balanceado**.
    - Ventajas: 
        - más simple, interpretable y más robusto para produccion.
    - Por desbalance del dataset, se priorizo **Recall/F1** de la clase minoritaria sobre accuracy.
- Se ejecutaron los `model-test` satisfactoriamente.

### API
- Se implementó el API con **FastAPI**.
- Se ejecutaron los `api-test` satisfactoriamente.

### Por qué balanceo de clases
- Se calcula `n_y0`, `n_y1`; y se usa como peso para **penalizar más** errores en `delay`.
- Aumenta el costo de fallar en la clase minoritaria y obliga al modelo a prestar atencion a eventos raros como `delay`.

### Cloud
- Se **dockerizó** el servicio para subir la imagen a **AWS ECR** y exponerla con **AWS App Runner**.
- El despliegue quedó **truncado** por limitaciones de **billing**.

### CI/CD
- Se implementó el pipeline de **CI/CD parcialmente**.
- No se pudo **validar al 100%** en **GitHub Actions** por limitaciones de **billing**.