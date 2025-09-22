import json
from google.cloud import bigquery
from google.oauth2 import service_account
import os

GCP_BUCKET_NAME= 'demanda-data'
GRANULARITY = 'NOMBRESUBDIRECCION' # 'PUNTOLOGISTICO'
DEEPGRANULARITY = 'NOMBREGRUPOESTADISTICO3' if GRANULARITY == 'NOMBRESUBDIRECCION' else 'SKU'
GRANULARITY_HISTORICO = 'NombreSubdireccion' # 'PuntoLogistico'
DEEPGRANULARITY_HISTORICO = 'NombreGrupoEstadistico3' if GRANULARITY_HISTORICO == 'NombreSubdireccion' else 'SKU'
ordenamiento_columnas = ['FECHA','NOMBREGRUPO','NOMBREDIRECCION','NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO1','NOMBREGRUPOESTADISTICO2','NOMBREGRUPOESTADISTICO3','MODEL','PRONOSTICO']

QUERY_RESULTADOS_MODEL2 = """
SELECT 
  `FECHA_EJECUCION`,
  `FECHA`,
  `COV%`,
  `MAPE%`,
  `MODEL`,
  `NOMBREGRUPOESTADISTICO3`,
  `NOMBRESUBDIRECCION`,
  `Y_HIST`,
  `Y_PREDICCION`,
  `Y_PRONOSTICOS`,
  `Y_TEST`
FROM demanda-prj-dev.pronosticos.forecast_table
WHERE FECHA_EJECUCION = (SELECT MAX( FECHA_EJECUCION ) FROM `demanda-prj-dev.pronosticos.forecast_table` WHERE TEST = '0')
"""

query_last_update_table = f"""
SELECT  
FORMAT_TIMESTAMP("%Y-%m-%d %H:%M:%S",CAST(FECHA_EJECUCION AS TIMESTAMP),"America/Monterrey") FECHA_EJECUCION,
CAST(TEST AS INT64) TEST,
IF( ( SELECT COUNT(DISTINCT MODEL) FROM `demanda-prj-dev.pronosticos.forecast_table` WHERE FECHA_EJECUCION = (SELECT MAX( FECHA_EJECUCION ) FROM `demanda-prj-dev.pronosticos.forecast_table`) ) = 1,MODEL,'Todos los modelos') MODEL,
FORECAST_TO_DO,
CAST(AVG(CAST(MONTHS_TO_FORECAST AS INT64)) AS INT64) MONTHS_TO_FORECAST,
CAST(AVG(CAST(MONTHS_TO_TEST AS INT64)) AS INT64) MONTHS_TO_TEST,
CAST(AVG(CAST(FROM_YEAR AS INT64)) AS INT64) FROM_YEAR,
COUNT(DISTINCT MODEL) TOTAL_MODELOS,
AVG(CAST(`MAPE_DEACERO%` AS FLOAT64)) MAPE_DEACERO,
APPROX_QUANTILES(CAST(`MAPE_DEACERO%` AS FLOAT64), 100)[OFFSET(15)] AS P15,
APPROX_QUANTILES(CAST(`MAPE_DEACERO%` AS FLOAT64), 100)[OFFSET(50)] AS P50,
APPROX_QUANTILES(CAST(`MAPE_DEACERO%` AS FLOAT64), 100)[OFFSET(85)] AS P85,
MIN(CAST(`MAPE_DEACERO%` AS FLOAT64)) AS MIN_MAPE_DEACERO,
MAX(CAST(`MAPE_DEACERO%` AS FLOAT64)) AS MAX_MAPE_DEACERO,
COUNT(DISTINCT CONCAT(NOMBRESUBDIRECCION,'-',NOMBREGRUPOESTADISTICO3) ) NUMERO_COMBINACIONES
FROM `demanda-prj-dev.pronosticos.forecast_table` 
WHERE FECHA_EJECUCION = (SELECT MAX( FECHA_EJECUCION ) FROM `demanda-prj-dev.pronosticos.forecast_table`)
GROUP BY FECHA_EJECUCION,CAST(TEST AS INT64),MODEL,FORECAST_TO_DO
"""


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "..\\..\\bd\\0_docs\\datahub-deacero-adojeda.json"

def data_bq(query):

    client = bigquery.Client(project="demanda-prj-dev")

    query = client.query(query)

    pv = query.to_dataframe(create_bqstorage_client=True)

    return pv

# RESULTADOS DE LA PRIMERA CONSULTA
results = data_bq(query_last_update_table)
print("RESULTADOS PRIMEER CONSULTA:")
display(results)

FROMYEAR = results.FROM_YEAR.values[0]
meses_validacion = results.MONTHS_TO_TEST.values[0]
meses_pronostico = results.MONTHS_TO_FORECAST.values[0]

print("PARAMETROS DE EJECUCION:")
print(FROMYEAR, meses_validacion,meses_pronostico )

query_mape_pvo_pvo_pronostico_evaluacion = f"""
WITH pvo_bp AS (
SELECT
  CASE 
    WHEN (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
         AND 
         DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)) THEN 'MES_VALIDACION'
    WHEN (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) 
         AND 
         DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_pronostico} MONTH)) THEN 'MES_PRONOSTICO'
  END 
  AS MES_TIPO,
  FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
  nom_subdireccion,
  nom_grupo_estadistico3,
  CASE WHEN SUM(`toneladas_pvo`) IS NULL THEN 0 ELSE SUM(`toneladas_pvo`) END toneladas_pvo,
  CASE WHEN SUM(`toneladas_plan_ventas`) IS NULL THEN 0 ELSE SUM(`toneladas_plan_ventas`) END toneladas_plan_ventas
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= {FROMYEAR}
  AND
   (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND 
    (`nom_grupo_estadistico1` IN ('VARILLA', 'MALLAS Y ALAMBRES',
      'SOLUCIONES', 'ALAMBRON', 'PERFILES',
      'INDUSTRIAL FILIALES', 'CABLES'))
  AND
  (
  (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
    AND 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH))
  OR
  (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN 
    DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) 
    AND 
    DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_pronostico} MONTH))
  )
  AND 
  nom_grupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
  AND
  nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
  AND
  nom_subdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
  AND
  nom_grupo_estadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
  AND
  nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
  AND
  nom_grupo_estadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
  -- Necesitamos que la informacion no sea nula
  AND nom_grupo IS NOT NULL
  AND nom_direccion IS NOT NULL
  AND nom_subdireccion IS NOT NULL
  AND nom_grupo_estadistico1 IS NOT NULL
  AND nom_grupo_estadistico2 IS NOT NULL
  AND nom_grupo_estadistico3 IS NOT NULL

GROUP BY
  MES_TIPO,
  mes_anio,
  nom_subdireccion,
  nom_grupo_estadistico3
),

tf AS (

SELECT
  FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
  nom_subdireccion,
  nom_grupo_estadistico3,
  CASE WHEN SUM(`toneladas_facturadas`) IS NULL THEN 0 ELSE SUM(`toneladas_facturadas`) END toneladas_facturadas,
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= {FROMYEAR}
  AND
   (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND 
    (`nom_grupo_estadistico1` IN ('VARILLA', 'MALLAS Y ALAMBRES',
      'SOLUCIONES', 'ALAMBRON', 'PERFILES',
      'INDUSTRIAL FILIALES', 'CABLES'))
  AND  
  PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
  AND 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)
  AND 
  nom_grupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
  AND
  nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
  AND
  nom_subdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
  AND
  nom_grupo_estadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
  AND
  nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
  AND
  nom_grupo_estadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
  -- Necesitamos que la informacion no sea nula
  AND nom_grupo IS NOT NULL
  AND nom_direccion IS NOT NULL
  AND nom_subdireccion IS NOT NULL
  AND nom_grupo_estadistico1 IS NOT NULL
  AND nom_grupo_estadistico2 IS NOT NULL
  AND nom_grupo_estadistico3 IS NOT NULL

GROUP BY
  mes_anio,
  nom_subdireccion,
  nom_grupo_estadistico3

),

mape_table AS (

SELECT  
pvo_bp.MES_TIPO,
pvo_bp.MesAnio,
pvo_bp.nom_subdireccion,
pvo_bp.nom_grupo_estadistico3,
pvo_bp.toneladas_pvo,
pvo_bp.toneladas_plan_ventas,
tf.toneladas_facturadas,
CASE 
  WHEN tf.toneladas_facturadas < 1 AND pvo_bp.toneladas_pvo < 1 THEN 0
  WHEN tf.toneladas_facturadas >= 1 AND pvo_bp.toneladas_pvo < 1 THEN 100
  WHEN tf.toneladas_facturadas < 1 AND pvo_bp.toneladas_pvo >= 1 THEN 100
  ELSE (ABS(tf.toneladas_facturadas-pvo_bp.toneladas_pvo)/pvo_bp.toneladas_pvo)*100
END AS MAPE_PVO,
CASE 
  WHEN tf.toneladas_facturadas < 1 AND pvo_bp.toneladas_plan_ventas < 1 THEN 0
  WHEN tf.toneladas_facturadas >= 1 AND pvo_bp.toneladas_plan_ventas < 1 THEN 100
  WHEN tf.toneladas_facturadas < 1 AND pvo_bp.toneladas_plan_ventas >= 1 THEN 100
  ELSE (ABS(tf.toneladas_facturadas-pvo_bp.toneladas_plan_ventas)/pvo_bp.toneladas_plan_ventas)*100
END AS MAPE_PV,
FROM pvo_bp LEFT JOIN tf 
ON pvo_bp.MesAnio = tf.MesAnio AND pvo_bp.nom_subdireccion = tf.nom_subdireccion AND pvo_bp.nom_grupo_estadistico3 = tf.nom_grupo_estadistico3

)

SELECT 
FORMAT_DATE('%Y-%m-%d',PARSE_DATE('%b-%Y', MesAnio)) AS FECHA,
nom_subdireccion AS NOMBRESUBDIRECCION,
nom_grupo_estadistico3 AS NOMBREGRUPOESTADISTICO3,
'PVO' AS MODELO_PVO,
'PV' AS MODELO_PV,
SUM(MAPE_PVO) AS PREDICCION_PVO,
SUM(MAPE_PV) AS PREDICCION_PV
FROM mape_table 
WHERE MES_TIPO = 'MES_VALIDACION'
GROUP BY 1,2,3,4,5
"""

query_mape_pvo_pv_validacion = f"""
WITH pvo_bp AS (
SELECT
  CASE 
    WHEN (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
         AND 
         DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)) THEN 'MES_VALIDACION'
    WHEN (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) 
         AND 
         DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_pronostico} MONTH)) THEN 'MES_PRONOSTICO'
  END 
  AS MES_TIPO,
  FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
  nom_subdireccion,
  nom_grupo_estadistico3,
  CASE WHEN SUM(`toneladas_pvo`) IS NULL THEN 0 ELSE SUM(`toneladas_pvo`) END toneladas_pvo,
  CASE WHEN SUM(`toneladas_plan_ventas`) IS NULL THEN 0 ELSE SUM(`toneladas_plan_ventas`) END toneladas_plan_ventas
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= {FROMYEAR}
  AND
   (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND 
    (`nom_grupo_estadistico1` IN ('VARILLA', 'MALLAS Y ALAMBRES',
      'SOLUCIONES', 'ALAMBRON', 'PERFILES',
      'INDUSTRIAL FILIALES', 'CABLES'))
  AND
  (
  (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
    AND 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH))
  OR
  (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN 
    DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) 
    AND 
    DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_pronostico} MONTH))
  )
  AND 
  nom_grupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
  AND
  nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
  AND
  nom_subdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
  AND
  nom_grupo_estadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
  AND
  nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
  AND
  nom_grupo_estadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
  -- Necesitamos que la informacion no sea nula
  AND nom_grupo IS NOT NULL
  AND nom_direccion IS NOT NULL
  AND nom_subdireccion IS NOT NULL
  AND nom_grupo_estadistico1 IS NOT NULL
  AND nom_grupo_estadistico2 IS NOT NULL
  AND nom_grupo_estadistico3 IS NOT NULL

GROUP BY
  MES_TIPO,
  mes_anio,
  nom_subdireccion,
  nom_grupo_estadistico3
),

tf AS (

SELECT
  FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
  nom_subdireccion,
  nom_grupo_estadistico3,
  CASE WHEN SUM(`toneladas_facturadas`) IS NULL THEN 0 ELSE SUM(`toneladas_facturadas`) END toneladas_facturadas,
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= {FROMYEAR}
  AND
   (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND 
    (`nom_grupo_estadistico1` IN ('VARILLA', 'MALLAS Y ALAMBRES',
      'SOLUCIONES', 'ALAMBRON', 'PERFILES',
      'INDUSTRIAL FILIALES', 'CABLES'))
  AND  
  PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
  AND 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)
  AND 
  nom_grupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
  AND
  nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
  AND
  nom_subdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
  AND
  nom_grupo_estadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
  AND
  nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
  AND
  nom_grupo_estadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
  -- Necesitamos que la informacion no sea nula
  AND nom_grupo IS NOT NULL
  AND nom_direccion IS NOT NULL
  AND nom_subdireccion IS NOT NULL
  AND nom_grupo_estadistico1 IS NOT NULL
  AND nom_grupo_estadistico2 IS NOT NULL
  AND nom_grupo_estadistico3 IS NOT NULL

GROUP BY
  mes_anio,
  nom_subdireccion,
  nom_grupo_estadistico3

),

mape_table AS (

SELECT  
pvo_bp.MES_TIPO,
pvo_bp.MesAnio,
pvo_bp.nom_subdireccion,
pvo_bp.nom_grupo_estadistico3,
pvo_bp.toneladas_pvo,
pvo_bp.toneladas_plan_ventas,
tf.toneladas_facturadas,
CASE 
  WHEN tf.toneladas_facturadas < 1 AND pvo_bp.toneladas_pvo < 1 THEN 0
  WHEN tf.toneladas_facturadas >= 1 AND pvo_bp.toneladas_pvo < 1 THEN 100
  WHEN tf.toneladas_facturadas < 1 AND pvo_bp.toneladas_pvo >= 1 THEN 100
  ELSE (ABS(tf.toneladas_facturadas-pvo_bp.toneladas_pvo)/pvo_bp.toneladas_pvo)*100
END AS MAPE_PVO,
CASE 
  WHEN tf.toneladas_facturadas < 1 AND pvo_bp.toneladas_plan_ventas < 1 THEN 0
  WHEN tf.toneladas_facturadas >= 1 AND pvo_bp.toneladas_plan_ventas < 1 THEN 100
  WHEN tf.toneladas_facturadas < 1 AND pvo_bp.toneladas_plan_ventas >= 1 THEN 100
  ELSE (ABS(tf.toneladas_facturadas-pvo_bp.toneladas_plan_ventas)/pvo_bp.toneladas_plan_ventas)*100
END AS MAPE_PV,
FROM pvo_bp LEFT JOIN tf 
ON pvo_bp.MesAnio = tf.MesAnio AND pvo_bp.nom_subdireccion = tf.nom_subdireccion AND pvo_bp.nom_grupo_estadistico3 = tf.nom_grupo_estadistico3

)

SELECT 
FORMAT_DATE('%Y-%m-%d',PARSE_DATE('%b-%Y', MesAnio)) AS FECHA,
nom_subdireccion AS NOMBRESUBDIRECCION,
nom_grupo_estadistico3 AS NOMBREGRUPOESTADISTICO3,
AVG(MAPE_PVO) AS MAPE_PVO,
AVG(MAPE_PV) AS MAPE_PV 
FROM mape_table 
WHERE MES_TIPO = 'MES_VALIDACION'
GROUP BY 1,2,3
"""

query_pvo_pv_pronostico = f"""
WITH pvo_bp AS (
SELECT
  CASE 
    WHEN (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
         AND 
         DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)) THEN 'MES_VALIDACION'
    WHEN (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) 
         AND 
         DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_pronostico} MONTH)) THEN 'MES_PRONOSTICO'
  END 
  AS MES_TIPO,
  FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
  nom_subdireccion,
  nom_grupo_estadistico3,
  CASE WHEN SUM(`toneladas_rolling_forecast`) IS NULL THEN 0 ELSE SUM(`toneladas_rolling_forecast`) END toneladas_pvo,
  CASE WHEN SUM(`toneladas_plan_ventas`) IS NULL THEN 0 ELSE SUM(`toneladas_plan_ventas`) END toneladas_plan_ventas
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= {FROMYEAR}
  AND
   (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND 
    (`nom_grupo_estadistico1` IN ('VARILLA', 'MALLAS Y ALAMBRES',
      'SOLUCIONES', 'ALAMBRON', 'PERFILES',
      'INDUSTRIAL FILIALES', 'CABLES'))
  AND
  (
  (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
    AND 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH))
  OR
  (PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN 
    DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) 
    AND 
    DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_pronostico} MONTH))
  )
  AND 
  nom_grupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
AND
nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
AND
nom_subdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
AND
nom_grupo_estadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
AND
nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
AND
nom_grupo_estadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
-- Necesitamos que la informacion no sea nula
AND nom_grupo IS NOT NULL
AND nom_direccion IS NOT NULL
AND nom_subdireccion IS NOT NULL
AND nom_grupo_estadistico1 IS NOT NULL
AND nom_grupo_estadistico2 IS NOT NULL
AND nom_grupo_estadistico3 IS NOT NULL

GROUP BY
  MES_TIPO,
  mes_anio,
  nom_subdireccion,
  nom_grupo_estadistico3
)

SELECT 
DISTINCT 
FORMAT_DATE('%Y-%m-%d',PARSE_DATE('%b-%Y', MesAnio)) AS FECHA,
nom_subdireccion AS NOMBRESUBDIRECCION,
nom_grupo_estadistico3 AS NOMBREGRUPOESTADISTICO3,
toneladas_pvo AS PVO,
toneladas_plan_ventas AS PV
FROM pvo_bp 
WHERE MES_TIPO = 'MES_PRONOSTICO'
"""

query_midas_pronostico_evaluacion = f"""
WITH models_results AS
(
SELECT DISTINCT FECHA,NOMBRESUBDIRECCION,NOMBREGRUPOESTADISTICO3,MODEL,CAST(`Y_PREDICCION` AS FLOAT64) AS PREDICCION_MIDAS,
FROM `demanda-prj-dev.pronosticos.forecast_table`
WHERE FECHA_EJECUCION = (SELECT MAX(FECHA_EJECUCION) FROM `demanda-prj-dev.pronosticos.forecast_table` WHERE TEST = '0')
)

SELECT * FROM models_results WHERE PREDICCION_MIDAS IS NOT NULL
"""

query_mape_midas_validacion = f"""
WITH models_results AS
(
SELECT DISTINCT FECHA,NOMBRESUBDIRECCION,NOMBREGRUPOESTADISTICO3,MODEL,CAST(`MAPE_DEACERO%` AS FLOAT64) AS MAPE_DEACERO,
FROM `demanda-prj-dev.pronosticos.forecast_table`
WHERE FECHA_EJECUCION = (SELECT MAX(FECHA_EJECUCION) FROM `demanda-prj-dev.pronosticos.forecast_table` WHERE TEST = '0')
),

ranking_table AS
(
SELECT FECHA,NOMBRESUBDIRECCION,NOMBREGRUPOESTADISTICO3,MODEL,MAPE_DEACERO,ROW_NUMBER() OVER (PARTITION BY FECHA,NOMBRESUBDIRECCION,NOMBREGRUPOESTADISTICO3 ORDER BY MAPE_DEACERO ASC) AS ranking
FROM models_results
)

SELECT 
CASE WHEN (PARSE_DATE('%Y-%m-%d', ranking_table.FECHA) BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
           AND 
         DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)) THEN ranking_table.FECHA 
END AS FECHA,
ranking_table.NOMBRESUBDIRECCION,
ranking_table.NOMBREGRUPOESTADISTICO3,
ranking_table.MODEL,
ranking_table.MAPE_DEACERO 
FROM ranking_table WHERE ranking = 1
AND CASE WHEN (PARSE_DATE('%Y-%m-%d', ranking_table.FECHA) BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) 
         AND 
         DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)) THEN ranking_table.FECHA END IS NOT NULL
"""

query_midas_pronostico = f"""
WITH models_forecast AS
(
SELECT DISTINCT FECHA,NOMBRESUBDIRECCION,NOMBREGRUPOESTADISTICO3,MODEL,CAST(Y_PRONOSTICOS AS FLOAT64) AS MIDAS,
FROM `demanda-prj-dev.pronosticos.forecast_table`
WHERE FECHA_EJECUCION = (SELECT MAX(FECHA_EJECUCION) FROM `demanda-prj-dev.pronosticos.forecast_table` WHERE TEST = '0')
AND Y_PRONOSTICOS IS NOT NULL
)

SELECT * FROM models_forecast
"""

query_historico_facturacion = f"""
with data_base AS (

SELECT

FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
nom_grupo AS NombreGrupo,
nom_direccion AS NombreDireccion,
nom_subdireccion AS NombreSubdireccion,
nom_grupo_estadistico1 AS NombreGrupoEstadistico1,
nom_grupo_estadistico2 AS NombreGrupoEstadistico2,
nom_grupo_estadistico3 AS NombreGrupoEstadistico3,
toneladas_facturadas AS Toneladas_Facturas
FROM `datahub-deacero.mart_comercial.comercial` 
WHERE EXTRACT(YEAR FROM fecha) >= {FROMYEAR} # Este es el que vamos a usar en productivo
AND nom_gerencia NOT IN ('ACERIAS SPOT')
AND 
  nom_grupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
AND
nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
AND
nom_subdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
AND
nom_grupo_estadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
AND
nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
AND
nom_grupo_estadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
-- Necesitamos que la informacion no sea nula
AND nom_grupo IS NOT NULL
AND nom_direccion IS NOT NULL
AND nom_subdireccion IS NOT NULL
AND nom_grupo_estadistico1 IS NOT NULL
AND nom_grupo_estadistico2 IS NOT NULL
AND nom_grupo_estadistico3 IS NOT NULL

)

# Agregar un cambio de valores para los cuales Toneladas_Facturas es 0 si el valor original es negativo

SELECT 
DISTINCT NombreGrupo,NombreDireccion,
NombreSubdireccion,NombreGrupoEstadistico1,
NombreGrupoEstadistico2,NombreGrupoEstadistico3,
FROM data_base
WHERE 
-- Descartamos los siguientes campos
NombreGrupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
AND
NombreDireccion NOT IN ('EXPORTACIÓN ALAMBRES')
AND
NombreSubdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
AND
NombreGrupoEstadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
AND
NombreGrupoEstadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
AND
NombreGrupoEstadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
-- Necesitamos que la informacion no sea nula
AND NombreGrupo IS NOT NULL
AND NombreDireccion IS NOT NULL
AND NombreSubdireccion IS NOT NULL
AND NombreGrupoEstadistico1 IS NOT NULL
AND NombreGrupoEstadistico2 IS NOT NULL
AND NombreGrupoEstadistico3 IS NOT NULL
GROUP BY
data_base.MesAnio,
data_base.NombreGrupo,
data_base.NombreDireccion,
data_base.NombreSubdireccion,
data_base.NombreGrupoEstadistico1,
data_base.NombreGrupoEstadistico2,
data_base.NombreGrupoEstadistico3

"""

query_historico_facturacion_evaluacion = f"""

with data_base AS (

SELECT

PARSE_DATE('%Y-%m-%d',FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%B.%Y', mes_anio))) AS FECHA,
nom_subdireccion AS NOMBRESUBDIRECCION,
nom_grupo_estadistico3 AS NOMBREGRUPOESTADISTICO3,
CASE WHEN SUM(`toneladas_facturadas`) IS NULL THEN 0
WHEN SUM(`toneladas_facturadas`) < 0 THEN 0 ELSE SUM(`toneladas_facturadas`) END TONELADAS_FACTURADAS
FROM `datahub-deacero.mart_comercial.comercial` 
WHERE EXTRACT(YEAR FROM fecha) >= {FROMYEAR} # Este es el que vamos a usar en productivo
AND nom_gerencia NOT IN ('ACERIAS SPOT')
AND 
  nom_grupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
AND
nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
AND
nom_subdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
AND
nom_grupo_estadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
AND
nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
AND
nom_grupo_estadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
-- Necesitamos que la informacion no sea nula
AND nom_grupo IS NOT NULL
AND nom_direccion IS NOT NULL
AND nom_subdireccion IS NOT NULL
AND nom_grupo_estadistico1 IS NOT NULL
AND nom_grupo_estadistico2 IS NOT NULL
AND nom_grupo_estadistico3 IS NOT NULL
GROUP BY 1,2,3

)

# Agregar un cambio de valores para los cuales Toneladas_Facturas es 0 si el valor original es negativo

SELECT 
DISTINCT 
FECHA,NOMBRESUBDIRECCION,NOMBREGRUPOESTADISTICO3,TONELADAS_FACTURADAS
FROM data_base
WHERE FECHA IS NOT NULL AND NOMBRESUBDIRECCION IS NOT NULL AND NOMBREGRUPOESTADISTICO3 IS NOT NULL AND TONELADAS_FACTURADAS IS NOT NULL
AND FECHA BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)
"""

query_avg_mape_max = f"""
SELECT 
FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%b-%Y', MesAnio)) FECHA,
AVG(MAPE_MAX) AVG_MAPE_MAX
FROM
(
SELECT
  DISTINCT
  FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
  nom_grupo,
  nom_direccion,
  nom_subdireccion,
  nom_grupo_estadistico1,
  nom_grupo_estadistico2,
  nom_grupo_estadistico3,
  #CASE WHEN SUM(`toneladas_mejor_pronostico`) IS NULL THEN 0 ELSE SUM(`toneladas_mejor_pronostico`) END toneladas_max, # pronostico
  #CASE WHEN SUM(`toneladas_facturadas`) IS NULL THEN 0 ELSE SUM(`toneladas_facturadas`) END toneladas_facturadas, # real
  LEAST( 
    200,
    CASE 
  WHEN SUM(IFNULL(toneladas_facturadas,0)) < 1 AND SUM(IFNULL(toneladas_mejor_pronostico,0)) < 1 THEN 0
  WHEN SUM(IFNULL(toneladas_facturadas,0)) >= 1 AND SUM(IFNULL(toneladas_mejor_pronostico,0)) < 1 THEN 100
  WHEN SUM(IFNULL(toneladas_facturadas,0)) < 1 AND SUM(IFNULL(toneladas_mejor_pronostico,0)) >= 1 THEN 100
  ELSE (ABS(SUM(IFNULL(toneladas_facturadas,0))-SUM(IFNULL(toneladas_mejor_pronostico,0)))/SUM(IFNULL(toneladas_mejor_pronostico,0)))*100
  END
   ) MAPE_MAX,
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= 2024
  AND
   (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND 
    (`nom_grupo_estadistico1` IN ('VARILLA', 'MALLAS Y ALAMBRES',
      'SOLUCIONES', 'ALAMBRON', 'PERFILES',
      'INDUSTRIAL FILIALES', 'CABLES'))
  AND 
  nom_grupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
AND
nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
AND
nom_subdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
AND
nom_grupo_estadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
AND
nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
AND
nom_grupo_estadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
-- Necesitamos que la informacion no sea nula
AND nom_grupo IS NOT NULL
AND nom_direccion IS NOT NULL
AND nom_subdireccion IS NOT NULL
AND nom_grupo_estadistico1 IS NOT NULL
AND nom_grupo_estadistico2 IS NOT NULL
AND nom_grupo_estadistico3 IS NOT NULL
  GROUP BY MesAnio,
nom_grupo,
nom_direccion,
nom_subdireccion,
nom_grupo_estadistico1,
nom_grupo_estadistico2,
nom_grupo_estadistico3
)
WHERE
  PARSE_DATE('%Y-%m-%d', FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%b-%Y', MesAnio))) <= DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)
  AND PARSE_DATE('%Y-%m-%d', FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%b-%Y', MesAnio))) IS NOT NULL
GROUP BY 1
"""

query_temporalidades_meses_pronostico_evaluacion = f"""
WITH db AS 
(
SELECT
  PARSE_DATE('%Y-%m-%d',FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%B.%Y', mes_anio))) AS MesAnio,
  nom_subdireccion,
  nom_grupo_estadistico3,
  CASE
    WHEN SUM(`toneladas_facturadas`) IS NULL THEN 0
    WHEN SUM(`toneladas_facturadas`) < 0 THEN 0
    ELSE SUM(`toneladas_facturadas`)
END
  toneladas_facturadas
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= {FROMYEAR}
  AND (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND (`nom_grupo_estadistico1` IN ('VARILLA',
      'MALLAS Y ALAMBRES',
      'SOLUCIONES',
      'ALAMBRON',
      'PERFILES',
      'INDUSTRIAL FILIALES',
      'CABLES'))
  AND nom_grupo NOT IN ('EMPRESAS RELACIONADAS',
    'TRASPASOS E INTEREMPRESAS')
  AND nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
  AND nom_subdireccion NOT IN ('EXCEDENTES ACEROS',
    'EXCEDENTES ALAMBRES',
    'RESTO ALAMBRES',
    'RESTO INGETEK')
  AND nom_grupo_estadistico1 NOT IN ('CHATARRA',
    'DEACERO POWER',
    'INTERNAS PRODUCCION ',
    'LOGÍSTICA',
    'PALANQUILLA',
    'SEGUNDAS')
  AND nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES',
    'CHQ')
  AND nom_grupo_estadistico3 NOT IN ('DESPERDICIO',
    'ALAMBRON SEGUNDAS',
    'ALAMBRON EXCEDENTES',
    'ALAMBRON TERCEROS',
    'ALAMBRON OTROS',
    'DERECHO DE VIA',
    'PILOTES',
    'PISO',
    'POLIZAS') -- Necesitamos que la informacion NO sea nula
  AND nom_grupo IS NOT NULL
  AND nom_direccion IS NOT NULL
  AND nom_subdireccion IS NOT NULL
  AND nom_grupo_estadistico1 IS NOT NULL
  AND nom_grupo_estadistico2 IS NOT NULL
  AND nom_grupo_estadistico3 IS NOT NULL

GROUP BY
  mes_anio,
  nom_subdireccion,
  nom_grupo_estadistico3
),

validacion_table_0 AS (

SELECT  
MesAnio,
nom_subdireccion,
nom_grupo_estadistico3,
AVG(toneladas_facturadas) OVER( PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING ) AS estimacion,
ROW_NUMBER() OVER(PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ) ranking
FROM db
WHERE MesAnio BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion}+6 MONTH) AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)

),

validacion_table AS (

SELECT MesAnio,nom_subdireccion,nom_grupo_estadistico3,estimacion
FROM validacion_table_0
WHERE ranking >= 7

),

pronostico_table_0 AS (

SELECT  
MesAnio,
nom_subdireccion,
nom_grupo_estadistico3,
AVG(toneladas_facturadas) OVER( PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING ) AS estimacion,
FROM db
WHERE MesAnio BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 6 MONTH) AND DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_pronostico} MONTH)

),

pronostico_table_1 AS (

SELECT  
MesAnio,
nom_subdireccion,
nom_grupo_estadistico3,
AVG(estimacion) OVER( PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING ) AS estimacion,
ROW_NUMBER() OVER(PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ) ranking
FROM pronostico_table_0

),

pronostico_table AS (

SELECT MesAnio,nom_subdireccion,nom_grupo_estadistico3,estimacion FROM pronostico_table_1 WHERE ranking >= 8

),

estimacion_table AS (

SELECT  
t1.MesAnio,
t1.nom_subdireccion,
t1.nom_grupo_estadistico3,
t1.estimacion
FROM pronostico_table AS t1

UNION ALL

SELECT  
t2.MesAnio,
t2.nom_subdireccion,
t2.nom_grupo_estadistico3,
t2.estimacion
FROM validacion_table AS t2

)

SELECT 
MesAnio AS FECHA,
nom_subdireccion AS NOMBRESUBDIRECCION,
nom_grupo_estadistico3 AS NOMBREGRUPOESTADISTICO3,
'TP_MESES' AS MODELO_TP_MESES,
estimacion AS PREDICCION_TP_MESES
FROM
(
SELECT 
ta.MesAnio,
ta.nom_subdireccion,
ta.nom_grupo_estadistico3,
ta.estimacion,
db.toneladas_facturadas,
CASE 
  WHEN ta.MesAnio BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)
  THEN SAFE_DIVIDE(ABS(db.toneladas_facturadas - ta.estimacion),ta.estimacion)*100 
  ELSE NULL
END MAPE,
CASE
  WHEN ta.MesAnio BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) THEN 'VALIDACION'
  ELSE 'PRONOSTICO'
END MES_TIPO
FROM estimacion_table AS ta
LEFT JOIN db
ON ta.MesAnio = db.MesAnio 
AND ta.nom_subdireccion = db.nom_subdireccion
AND ta.nom_grupo_estadistico3 = db.nom_grupo_estadistico3
)
WHERE MES_TIPO = 'VALIDACION'
"""

query_temporalidades_meses = f"""
WITH db AS 
(
SELECT
  PARSE_DATE('%Y-%m-%d',FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%B.%Y', mes_anio))) AS MesAnio,
  nom_subdireccion,
  nom_grupo_estadistico3,
  CASE
    WHEN SUM(`toneladas_facturadas`) IS NULL THEN 0
    WHEN SUM(`toneladas_facturadas`) < 0 THEN 0
    ELSE SUM(`toneladas_facturadas`)
END
  toneladas_facturadas
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= {FROMYEAR}
  AND (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND (`nom_grupo_estadistico1` IN ('VARILLA',
      'MALLAS Y ALAMBRES',
      'SOLUCIONES',
      'ALAMBRON',
      'PERFILES',
      'INDUSTRIAL FILIALES',
      'CABLES'))
  AND nom_grupo NOT IN ('EMPRESAS RELACIONADAS',
    'TRASPASOS E INTEREMPRESAS')
  AND nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
  AND nom_subdireccion NOT IN ('EXCEDENTES ACEROS',
    'EXCEDENTES ALAMBRES',
    'RESTO ALAMBRES',
    'RESTO INGETEK')
  AND nom_grupo_estadistico1 NOT IN ('CHATARRA',
    'DEACERO POWER',
    'INTERNAS PRODUCCION ',
    'LOGÍSTICA',
    'PALANQUILLA',
    'SEGUNDAS')
  AND nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES',
    'CHQ')
  AND nom_grupo_estadistico3 NOT IN ('DESPERDICIO',
    'ALAMBRON SEGUNDAS',
    'ALAMBRON EXCEDENTES',
    'ALAMBRON TERCEROS',
    'ALAMBRON OTROS',
    'DERECHO DE VIA',
    'PILOTES',
    'PISO',
    'POLIZAS') -- Necesitamos que la informacion NO sea nula
  AND nom_grupo IS NOT NULL
  AND nom_direccion IS NOT NULL
  AND nom_subdireccion IS NOT NULL
  AND nom_grupo_estadistico1 IS NOT NULL
  AND nom_grupo_estadistico2 IS NOT NULL
  AND nom_grupo_estadistico3 IS NOT NULL

GROUP BY
  mes_anio,
  nom_subdireccion,
  nom_grupo_estadistico3
),

validacion_table_0 AS (

SELECT  
MesAnio,
nom_subdireccion,
nom_grupo_estadistico3,
AVG(toneladas_facturadas) OVER( PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING ) AS estimacion,
ROW_NUMBER() OVER(PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ) ranking
FROM db
WHERE MesAnio BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion}+6 MONTH) AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)

),

validacion_table AS (

SELECT MesAnio,nom_subdireccion,nom_grupo_estadistico3,estimacion
FROM validacion_table_0
WHERE ranking >= 7

),

pronostico_table_0 AS (

SELECT  
MesAnio,
nom_subdireccion,
nom_grupo_estadistico3,
AVG(toneladas_facturadas) OVER( PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING ) AS estimacion,
FROM db
WHERE MesAnio BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 6 MONTH) AND DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_pronostico} MONTH)

),

pronostico_table_1 AS (

SELECT  
MesAnio,
nom_subdireccion,
nom_grupo_estadistico3,
AVG(estimacion) OVER( PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING ) AS estimacion,
ROW_NUMBER() OVER(PARTITION BY nom_subdireccion,nom_grupo_estadistico3 ORDER BY MesAnio ASC ) ranking
FROM pronostico_table_0

),

pronostico_table AS (

SELECT MesAnio,nom_subdireccion,nom_grupo_estadistico3,estimacion FROM pronostico_table_1 WHERE ranking >= 8

),

estimacion_table AS (

SELECT  
t1.MesAnio,
t1.nom_subdireccion,
t1.nom_grupo_estadistico3,
t1.estimacion
FROM pronostico_table AS t1

UNION ALL

SELECT  
t2.MesAnio,
t2.nom_subdireccion,
t2.nom_grupo_estadistico3,
t2.estimacion
FROM validacion_table AS t2

)

SELECT 
MesAnio AS FECHA,
nom_subdireccion AS NOMBRESUBDIRECCION,
nom_grupo_estadistico3 AS NOMBREGRUPOESTADISTICO3,
estimacion AS TP_MESES,
toneladas_facturadas AS TONELADAS_FACTURADAS,
CASE 
  WHEN toneladas_facturadas < 1 AND estimacion < 1 AND MES_TIPO = 'VALIDACION' THEN 0
  WHEN toneladas_facturadas >= 1 AND estimacion < 1 AND MES_TIPO = 'VALIDACION' THEN 100
  WHEN toneladas_facturadas < 1 AND estimacion >= 1 AND MES_TIPO = 'VALIDACION' THEN 100
  ELSE MAPE
END MAPE_TEMPORALIDADES_MESES,
MES_TIPO
FROM
(
SELECT 
ta.MesAnio,
ta.nom_subdireccion,
ta.nom_grupo_estadistico3,
ta.estimacion,
db.toneladas_facturadas,
CASE 
  WHEN ta.MesAnio BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)
  THEN SAFE_DIVIDE(ABS(db.toneladas_facturadas - ta.estimacion),ta.estimacion)*100 
  ELSE NULL
END MAPE,
CASE
  WHEN ta.MesAnio BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) THEN 'VALIDACION'
  ELSE 'PRONOSTICO'
END MES_TIPO
FROM estimacion_table AS ta
LEFT JOIN db
ON ta.MesAnio = db.MesAnio 
AND ta.nom_subdireccion = db.nom_subdireccion
AND ta.nom_grupo_estadistico3 = db.nom_grupo_estadistico3
)
"""

query_temporalidades_movil = f"""
WITH db AS 
(
SELECT
  PARSE_DATE('%Y-%m-%d',FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%B.%Y', mes_anio))) AS MesAnio,
  nom_subdireccion,
  nom_grupo_estadistico3,
  CASE
    WHEN SUM(`toneladas_facturadas`) IS NULL THEN 0
    WHEN SUM(`toneladas_facturadas`) < 0 THEN 0
    ELSE SUM(`toneladas_facturadas`)
END
  toneladas_facturadas
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= ( SELECT EXTRACT(YEAR FROM CURRENT_DATE()) - 6 )
  AND (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND (`nom_grupo_estadistico1` IN ('VARILLA',
      'MALLAS Y ALAMBRES',
      'SOLUCIONES',
      'ALAMBRON',
      'PERFILES',
      'INDUSTRIAL FILIALES',
      'CABLES'))
  AND nom_grupo NOT IN ('EMPRESAS RELACIONADAS',
    'TRASPASOS E INTEREMPRESAS')
  AND nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
  AND nom_subdireccion NOT IN ('EXCEDENTES ACEROS',
    'EXCEDENTES ALAMBRES',
    'RESTO ALAMBRES',
    'RESTO INGETEK')
  AND nom_grupo_estadistico1 NOT IN ('CHATARRA',
    'DEACERO POWER',
    'INTERNAS PRODUCCION ',
    'LOGÍSTICA',
    'PALANQUILLA',
    'SEGUNDAS')
  AND nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES',
    'CHQ')
  AND nom_grupo_estadistico3 NOT IN ('DESPERDICIO',
    'ALAMBRON SEGUNDAS',
    'ALAMBRON EXCEDENTES',
    'ALAMBRON TERCEROS',
    'ALAMBRON OTROS',
    'DERECHO DE VIA',
    'PILOTES',
    'PISO',
    'POLIZAS') 
  AND nom_grupo IS NOT NULL
  AND nom_direccion IS NOT NULL
  AND nom_subdireccion IS NOT NULL
  AND nom_grupo_estadistico1 IS NOT NULL
  AND nom_grupo_estadistico2 IS NOT NULL
  AND nom_grupo_estadistico3 IS NOT NULL

GROUP BY
  mes_anio,
  nom_subdireccion,
  nom_grupo_estadistico3
),

validacion_table_0 AS (

SELECT  
MesAnio AS FECHA,
DATE_SUB(MesAnio, INTERVAL 1 YEAR) FECHA_1_ANIO,
DATE_SUB(MesAnio, INTERVAL 2 YEAR) FECHA_2_ANIO,
DATE_SUB(MesAnio, INTERVAL 3 YEAR) FECHA_3_ANIO,
DATE_SUB(MesAnio, INTERVAL 4 YEAR) FECHA_4_ANIO,
DATE_SUB(MesAnio, INTERVAL 5 YEAR) FECHA_5_ANIO,
DATE_SUB(MesAnio, INTERVAL 6 YEAR) FECHA_6_ANIO,
nom_subdireccion,
nom_grupo_estadistico3,
toneladas_facturadas
FROM db

-- Aqui el 6 es el numero de meses exacto de evaluacion

WHERE MesAnio BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_validacion} MONTH) AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)

),

validacion_table_1 AS (

SELECT 
validacion_table_0.FECHA,
validacion_table_0.FECHA_1_ANIO,
validacion_table_0.FECHA_2_ANIO,
validacion_table_0.FECHA_3_ANIO,
validacion_table_0.FECHA_4_ANIO,
validacion_table_0.FECHA_5_ANIO,
validacion_table_0.FECHA_6_ANIO,
validacion_table_0.nom_subdireccion,
validacion_table_0.nom_grupo_estadistico3,
validacion_table_0.toneladas_facturadas,
IFNULL(t0.toneladas_facturadas,0) AS toneladas_facturadas_1_ANIO,
IFNULL(t1.toneladas_facturadas,0) AS toneladas_facturadas_2_ANIO,
IFNULL(t2.toneladas_facturadas,0) AS toneladas_facturadas_3_ANIO,
IFNULL(t3.toneladas_facturadas,0) AS toneladas_facturadas_4_ANIO,
IFNULL(t4.toneladas_facturadas,0) AS toneladas_facturadas_5_ANIO,
IFNULL(t5.toneladas_facturadas,0) AS toneladas_facturadas_6_ANIO
FROM validacion_table_0
LEFT JOIN db AS t0
ON validacion_table_0.FECHA_1_ANIO = t0.MesAnio AND validacion_table_0.nom_subdireccion = t0.nom_subdireccion AND validacion_table_0.nom_grupo_estadistico3 = t0.nom_grupo_estadistico3
LEFT JOIN db AS t1 
ON validacion_table_0.FECHA_2_ANIO = t1.MesAnio AND validacion_table_0.nom_subdireccion = t1.nom_subdireccion AND validacion_table_0.nom_grupo_estadistico3 = t1.nom_grupo_estadistico3
LEFT JOIN db AS t2
ON validacion_table_0.FECHA_3_ANIO = t2.MesAnio AND validacion_table_0.nom_subdireccion = t2.nom_subdireccion AND validacion_table_0.nom_grupo_estadistico3 = t2.nom_grupo_estadistico3
LEFT JOIN db AS t3
ON validacion_table_0.FECHA_4_ANIO = t3.MesAnio AND validacion_table_0.nom_subdireccion = t3.nom_subdireccion AND validacion_table_0.nom_grupo_estadistico3 = t3.nom_grupo_estadistico3
LEFT JOIN db AS t4
ON validacion_table_0.FECHA_5_ANIO = t4.MesAnio AND validacion_table_0.nom_subdireccion = t4.nom_subdireccion AND validacion_table_0.nom_grupo_estadistico3 = t4.nom_grupo_estadistico3
LEFT JOIN db AS t5
ON validacion_table_0.FECHA_6_ANIO = t5.MesAnio AND validacion_table_0.nom_subdireccion = t5.nom_subdireccion AND validacion_table_0.nom_grupo_estadistico3 = t5.nom_grupo_estadistico3
),

validacion_table AS (

SELECT 
FECHA,
nom_subdireccion,
nom_grupo_estadistico3,
toneladas_facturadas,
(IFNULL(toneladas_facturadas_1_ANIO, 0) + IFNULL(toneladas_facturadas_2_ANIO, 0) + IFNULL(toneladas_facturadas_3_ANIO, 0) + IFNULL(toneladas_facturadas_4_ANIO,0) + IFNULL(toneladas_facturadas_5_ANIO,0) + IFNULL(toneladas_facturadas_6_ANIO,0)) / 
    NULLIF((CASE WHEN toneladas_facturadas_1_ANIO IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN toneladas_facturadas_2_ANIO IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN toneladas_facturadas_3_ANIO IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN toneladas_facturadas_4_ANIO IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN toneladas_facturadas_5_ANIO IS NOT NULL THEN 1 ELSE 0 END + 
            CASE WHEN toneladas_facturadas_6_ANIO IS NOT NULL THEN 1 ELSE 0 END
            ), 0) estimacion
FROM validacion_table_1

),

pronostico_table_0 AS (


SELECT  
MesAnio AS FECHA,
DATE_SUB(MesAnio, INTERVAL 1 YEAR) FECHA_1_ANIO,
DATE_SUB(MesAnio, INTERVAL 2 YEAR) FECHA_2_ANIO,
DATE_SUB(MesAnio, INTERVAL 3 YEAR) FECHA_3_ANIO,
DATE_SUB(MesAnio, INTERVAL 4 YEAR) FECHA_4_ANIO,
DATE_SUB(MesAnio, INTERVAL 5 YEAR) FECHA_5_ANIO,
DATE_SUB(MesAnio, INTERVAL 6 YEAR) FECHA_6_ANIO,
nom_subdireccion,
nom_grupo_estadistico3,
toneladas_facturadas
FROM db

-- Aqui el 6 es el numero de meses exacto de pronostico

WHERE MesAnio BETWEEN DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) AND DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL {meses_pronostico} MONTH)

),

pronostico_table_1 AS (

SELECT 
pronostico_table_0.FECHA,
pronostico_table_0.FECHA_1_ANIO,
pronostico_table_0.FECHA_2_ANIO,
pronostico_table_0.FECHA_3_ANIO,
pronostico_table_0.FECHA_4_ANIO,
pronostico_table_0.FECHA_5_ANIO,
pronostico_table_0.FECHA_6_ANIO,
pronostico_table_0.nom_subdireccion,
pronostico_table_0.nom_grupo_estadistico3,
pronostico_table_0.toneladas_facturadas,
IFNULL(t0.toneladas_facturadas,0) AS toneladas_facturadas_1_ANIO,
IFNULL(t1.toneladas_facturadas,0) AS toneladas_facturadas_2_ANIO,
IFNULL(t2.toneladas_facturadas,0) AS toneladas_facturadas_3_ANIO,
IFNULL(t3.toneladas_facturadas,0) AS toneladas_facturadas_4_ANIO,
IFNULL(t4.toneladas_facturadas,0) AS toneladas_facturadas_5_ANIO,
IFNULL(t5.toneladas_facturadas,0) AS toneladas_facturadas_6_ANIO
FROM pronostico_table_0
LEFT JOIN db AS t0
ON pronostico_table_0.FECHA_1_ANIO = t0.MesAnio AND pronostico_table_0.nom_subdireccion = t0.nom_subdireccion AND pronostico_table_0.nom_grupo_estadistico3 = t0.nom_grupo_estadistico3
LEFT JOIN db AS t1 
ON pronostico_table_0.FECHA_2_ANIO = t1.MesAnio AND pronostico_table_0.nom_subdireccion = t1.nom_subdireccion AND pronostico_table_0.nom_grupo_estadistico3 = t1.nom_grupo_estadistico3
LEFT JOIN db AS t2
ON pronostico_table_0.FECHA_3_ANIO = t2.MesAnio AND pronostico_table_0.nom_subdireccion = t2.nom_subdireccion AND pronostico_table_0.nom_grupo_estadistico3 = t2.nom_grupo_estadistico3
LEFT JOIN db AS t3
ON pronostico_table_0.FECHA_4_ANIO = t3.MesAnio AND pronostico_table_0.nom_subdireccion = t3.nom_subdireccion AND pronostico_table_0.nom_grupo_estadistico3 = t3.nom_grupo_estadistico3
LEFT JOIN db AS t4
ON pronostico_table_0.FECHA_5_ANIO = t4.MesAnio AND pronostico_table_0.nom_subdireccion = t4.nom_subdireccion AND pronostico_table_0.nom_grupo_estadistico3 = t4.nom_grupo_estadistico3
LEFT JOIN db AS t5
ON pronostico_table_0
.FECHA_6_ANIO = t5.MesAnio AND pronostico_table_0.nom_subdireccion = t5.nom_subdireccion AND pronostico_table_0.nom_grupo_estadistico3 = t5.nom_grupo_estadistico3

),

pronostico_table AS (

SELECT 
FECHA,
nom_subdireccion,
nom_grupo_estadistico3,
toneladas_facturadas,
(IFNULL(toneladas_facturadas_1_ANIO, 0) + IFNULL(toneladas_facturadas_2_ANIO, 0) + IFNULL(toneladas_facturadas_3_ANIO, 0) + IFNULL(toneladas_facturadas_4_ANIO,0) + IFNULL(toneladas_facturadas_5_ANIO,0) + IFNULL(toneladas_facturadas_6_ANIO,0)) / 
    NULLIF((CASE WHEN toneladas_facturadas_1_ANIO IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN toneladas_facturadas_2_ANIO IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN toneladas_facturadas_3_ANIO IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN toneladas_facturadas_4_ANIO IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN toneladas_facturadas_5_ANIO IS NOT NULL THEN 1 ELSE 0 END + 
            CASE WHEN toneladas_facturadas_6_ANIO IS NOT NULL THEN 1 ELSE 0 END
            ), 0) estimacion
FROM pronostico_table_1

),

estimacion_table AS (

SELECT 
t1.FECHA,
t1.nom_subdireccion,
t1.nom_grupo_estadistico3,
t1.toneladas_facturadas,
t1.estimacion,
'VALIDACION' MES_TIPO
FROM validacion_table AS t1

UNION ALL

SELECT 
t2.FECHA,
t2.nom_subdireccion,
t2.nom_grupo_estadistico3,
t2.toneladas_facturadas,
t2.estimacion,
'PRONOSTICO' MES_TIPO
FROM pronostico_table AS t2


)

SELECT 
FECHA,
nom_subdireccion AS NOMBRESUBDIRECCION,
nom_grupo_estadistico3 AS NOMBREGRUPOESTADISTICO3,
toneladas_facturadas AS TONELADAS_FACTURADAS,
estimacion AS TP_MOVIL,
CASE
  WHEN toneladas_facturadas < 1 AND estimacion < 1 AND MES_TIPO = 'VALIDACION' THEN 0
  WHEN toneladas_facturadas >= 1 AND estimacion < 1 AND MES_TIPO = 'VALIDACION' THEN 100
  WHEN toneladas_facturadas < 1 AND estimacion >= 1 AND MES_TIPO = 'VALIDACION' THEN 100
  WHEN MES_TIPO = 'PRONOSTICO' THEN NULL
  ELSE SAFE_DIVIDE(ABS(toneladas_facturadas-estimacion),estimacion)*100
END MAPE_TEMPORALIDADES_MOVIL,
MES_TIPO
FROM estimacion_table
"""

query_demand_management = f"""
SELECT * FROM `demanda-prj-dev.pronosticos.Pronostico_DemandManagement`
"""

query_demand_management_pronostico = f"""
SELECT 
FECHA, 
NOMBREGRUPOESTADISTICO3, 
NOMBRESUBDIRECCION, 
PRONOSTICO AS DM
FROM `demanda-prj-dev.pronosticos.Pronostico_DemandManagement`
WHERE PARSE_DATE('%Y-%m-%d',FECHA) BETWEEN DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) AND DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 6 MONTH)
"""

query_toneladas_facturadas_dm = f"""
SELECT
  FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%B.%Y', mes_anio)) AS FECHA,
  nom_subdireccion NOMBRESUBDIRECCION,
  nom_grupo_estadistico3 NOMBREGRUPOESTADISTICO3,
  CASE WHEN SUM(`toneladas_facturadas`) IS NULL THEN 0 ELSE SUM(`toneladas_facturadas`) END TONELADAS_FACTURADAS,
FROM
  `datahub-deacero.mart_comercial.comercial`
WHERE
  `anio` >= 2025
  AND
   (`nom_grupo` IN ('ACEROS',
      'ALAMBRES',
      'DEACERO SOLUTIONS',
      'FILIALES ALAMBRES',
      'USA'))
  AND 
    (`nom_grupo_estadistico1` IN ('VARILLA', 'MALLAS Y ALAMBRES',
      'SOLUCIONES', 'ALAMBRON', 'PERFILES',
      'INDUSTRIAL FILIALES', 'CABLES'))
  AND  
  PARSE_DATE('%b-%Y', FORMAT_DATE('%b-%Y', PARSE_DATE('%B.%Y', mes_anio))) BETWEEN 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 6 MONTH) 
  AND 
    DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)
  AND 
  nom_grupo NOT IN ('EMPRESAS RELACIONADAS','TRASPASOS E INTEREMPRESAS')
  AND
  nom_direccion NOT IN ('EXPORTACIÓN ALAMBRES')
  AND
  nom_subdireccion NOT IN ('EXCEDENTES ACEROS','EXCEDENTES ALAMBRES','RESTO ALAMBRES','RESTO INGETEK')
  AND
  nom_grupo_estadistico1 NOT IN ('CHATARRA','DEACERO POWER','INTERNAS PRODUCCION ','LOGÍSTICA','PALANQUILLA','SEGUNDAS')
  AND
  nom_grupo_estadistico2 NOT IN ('SEGUNDAS / EXCEDENTES','CHQ')
  AND
  nom_grupo_estadistico3 NOT IN ('DESPERDICIO','ALAMBRON SEGUNDAS','ALAMBRON EXCEDENTES','ALAMBRON TERCEROS','ALAMBRON OTROS','DERECHO DE VIA','PILOTES','PISO','POLIZAS')
  -- Necesitamos que la informacion no sea nula
  AND nom_grupo IS NOT NULL
  AND nom_direccion IS NOT NULL
  AND nom_subdireccion IS NOT NULL
  AND nom_grupo_estadistico1 IS NOT NULL
  AND nom_grupo_estadistico2 IS NOT NULL
  AND nom_grupo_estadistico3 IS NOT NULL

GROUP BY
  mes_anio,
  nom_subdireccion,
  nom_grupo_estadistico3
"""

QUERY_HISTORICO_FACTURACION = f"""
with data_base AS (

SELECT

FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
nom_grupo AS NombreGrupo,
nom_direccion AS NombreDireccion,
nom_subdireccion AS NombreSubdireccion,
nom_grupo_estadistico1 AS NombreGrupoEstadistico1,
nom_grupo_estadistico2 AS NombreGrupoEstadistico2,
nom_grupo_estadistico3 AS NombreGrupoEstadistico3,
toneladas_facturadas AS toneladas_facturadas
FROM `datahub-deacero.mart_comercial.comercial` 
WHERE EXTRACT(YEAR FROM fecha) >= 2022 # Este es el que vamos a usar en productivo
)

# Agregar un cambio de valores para los cuales toneladas_facturadas es 0 si el valor original es negativo

SELECT 
MesAnio,NombreGrupo,NombreDireccion,
NombreSubdireccion,NombreGrupoEstadistico1,
NombreGrupoEstadistico2,NombreGrupoEstadistico3,
CASE 
  WHEN toneladas_facturadas IS NULL THEN 0
  WHEN toneladas_facturadas < 0 THEN 0
  WHEN toneladas_facturadas >= 0 THEN toneladas_facturadas
END toneladas_facturadas

FROM
(
SELECT 
data_base.MesAnio,
data_base.NombreGrupo,
data_base.NombreDireccion,
data_base.NombreSubdireccion,
data_base.NombreGrupoEstadistico1,
data_base.NombreGrupoEstadistico2,
data_base.NombreGrupoEstadistico3,
SUM(data_base.toneladas_facturadas) AS toneladas_facturadas
FROM data_base
WHERE MesAnio IS NOT NULL
AND NombreGrupo IS NOT NULL
AND NombreDireccion IS NOT NULL
AND NombreSubdireccion IS NOT NULL
AND NombreGrupoEstadistico1 IS NOT NULL
AND NombreGrupoEstadistico2 IS NOT NULL
AND NombreGrupoEstadistico3 IS NOT NULL
GROUP BY
data_base.MesAnio,
data_base.NombreGrupo,
data_base.NombreDireccion,
data_base.NombreSubdireccion,
data_base.NombreGrupoEstadistico1,
data_base.NombreGrupoEstadistico2,
data_base.NombreGrupoEstadistico3
)
WHERE MesAnio <= '2026-02-01'
ORDER BY MesAnio ASC, toneladas_facturadas DESC
"""

QUERY_HISTORICO_PVO = f"""
with data_base AS (
SELECT
FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
nom_grupo AS NombreGrupo,
nom_direccion AS NombreDireccion,
nom_subdireccion AS NombreSubdireccion,
nom_grupo_estadistico1 AS NombreGrupoEstadistico1,
nom_grupo_estadistico2 AS NombreGrupoEstadistico2,
nom_grupo_estadistico3 AS NombreGrupoEstadistico3,
toneladas_pvo AS toneladas_pvo
FROM `datahub-deacero.mart_comercial.comercial` 
WHERE EXTRACT(YEAR FROM fecha) >= 2022 # Este es el que vamos a usar en productivo
)
# Agregar un cambio de valores para los cuales toneladas_pvo es 0 si el valor original es negativo

SELECT 
MesAnio,NombreGrupo,NombreDireccion,
NombreSubdireccion,NombreGrupoEstadistico1,
NombreGrupoEstadistico2,NombreGrupoEstadistico3,
CASE 
  WHEN toneladas_pvo >= 0 THEN toneladas_pvo
  WHEN toneladas_pvo < 0 THEN toneladas_pvo
END toneladas_pvo
FROM
(
SELECT 
data_base.MesAnio,
data_base.NombreGrupo,
data_base.NombreDireccion,
data_base.NombreSubdireccion,
data_base.NombreGrupoEstadistico1,
data_base.NombreGrupoEstadistico2,
data_base.NombreGrupoEstadistico3,
SUM(data_base.toneladas_pvo) AS toneladas_pvo
FROM data_base
-- Necesitamos que la informacion no sea nula
WHERE MesAnio IS NOT NULL
AND NombreGrupo IS NOT NULL
AND NombreDireccion IS NOT NULL
AND NombreSubdireccion IS NOT NULL
AND NombreGrupoEstadistico1 IS NOT NULL
AND NombreGrupoEstadistico2 IS NOT NULL
AND NombreGrupoEstadistico3 IS NOT NULL
GROUP BY
data_base.MesAnio,
data_base.NombreGrupo,
data_base.NombreDireccion,
data_base.NombreSubdireccion,
data_base.NombreGrupoEstadistico1,
data_base.NombreGrupoEstadistico2,
data_base.NombreGrupoEstadistico3
)
WHERE MesAnio <= '2026-02-01'
ORDER BY MesAnio ASC, toneladas_pvo DESC
"""

QUERY_HISTORICO_PV = f"""
with data_base AS (

SELECT

FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%B.%Y', mes_anio)) AS MesAnio,
nom_grupo AS NombreGrupo,
nom_direccion AS NombreDireccion,
nom_subdireccion AS NombreSubdireccion,
nom_grupo_estadistico1 AS NombreGrupoEstadistico1,
nom_grupo_estadistico2 AS NombreGrupoEstadistico2,
nom_grupo_estadistico3 AS NombreGrupoEstadistico3,
toneladas_plan_ventas AS toneladas_plan_ventas
FROM `datahub-deacero.mart_comercial.comercial` 
WHERE EXTRACT(YEAR FROM fecha) >= 2022 # Este es el que vamos a usar en productivo
#AND nom_gerencia NOT IN ('ACERIAS SPOT')
)
# Agregar un cambio de valores para los cuales toneladas_plan_ventas es 0 si el valor original es negativo

SELECT 
MesAnio,NombreGrupo,NombreDireccion,
NombreSubdireccion,NombreGrupoEstadistico1,
NombreGrupoEstadistico2,NombreGrupoEstadistico3,
CASE 
  WHEN toneladas_plan_ventas >= 0 THEN toneladas_plan_ventas
  WHEN toneladas_plan_ventas < 0 THEN toneladas_plan_ventas
END toneladas_plan_ventas
FROM
(
SELECT 
data_base.MesAnio,
data_base.NombreGrupo,
data_base.NombreDireccion,
data_base.NombreSubdireccion,
data_base.NombreGrupoEstadistico1,
data_base.NombreGrupoEstadistico2,
data_base.NombreGrupoEstadistico3,
SUM(data_base.toneladas_plan_ventas) AS toneladas_plan_ventas
FROM data_base
WHERE MesAnio IS NOT NULL
AND NombreGrupo IS NOT NULL
AND NombreDireccion IS NOT NULL
AND NombreSubdireccion IS NOT NULL
AND NombreGrupoEstadistico1 IS NOT NULL
AND NombreGrupoEstadistico2 IS NOT NULL
AND NombreGrupoEstadistico3 IS NOT NULL
GROUP BY
data_base.MesAnio,
data_base.NombreGrupo,
data_base.NombreDireccion,
data_base.NombreSubdireccion,
data_base.NombreGrupoEstadistico1,
data_base.NombreGrupoEstadistico2,
data_base.NombreGrupoEstadistico3
)
WHERE MesAnio <= '2026-02-01'
# WHERE MesAnio < FORMAT_DATE('%Y-%m-%d', DATE_TRUNC(CURRENT_DATE(), MONTH))
ORDER BY MesAnio ASC, toneladas_plan_ventas DESC
"""

