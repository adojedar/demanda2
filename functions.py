from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import Conflict,NotFound
import pandas as pd
import numpy as np 
import json
from constants import *
import os
from dateutil.relativedelta import relativedelta
import itertools
import datetime
import pytz
import warnings
warnings.filterwarnings('ignore')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCIONES POP
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def transform_MIDAS(df):
    # --- Define columns to extract ---
    colNames = ["FECHA", "Y_PREDICCION", "NOMBRESUBDIRECCION", "NOMBREGRUPOESTADISTICO3", "MODEL"]
    colNames2 = ["FECHA", "Y_PRONOSTICOS", "NOMBRESUBDIRECCION", "NOMBREGRUPOESTADISTICO3", "MODEL"]
    colNames3 = ["FECHA", "Y_HIST", "NOMBRESUBDIRECCION", "NOMBREGRUPOESTADISTICO3"]

    df1 = df[colNames].copy() #df["Y_PREDICCION"].notna()]
    df1["Y_PREDICCION"] = df1["Y_PREDICCION"].astype(float) 
    df2 = df[colNames2].copy() #df["Y_PRONOSTICOS"].notna()
    df2["Y_PRONOSTICOS"] = df2["Y_PRONOSTICOS"].astype(float)
    df2.columns = colNames  
    df3 = df[df.MODEL == "BayesianRidge"][colNames3].copy()
    df3["Y_HIST"] = df3["Y_HIST"].astype(float)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    # --- Reorder columns: move Y_PREDICCION next to MODEL ---
    cols = combined_df.columns.tolist()
    y_pred_idx = cols.index("Y_PREDICCION")
    model_idx = cols.index("MODEL")
    cols.insert(model_idx + 1, cols.pop(y_pred_idx))
    combined_df = combined_df[cols]
    combined_df["Y_PREDICCION"] = combined_df["Y_PREDICCION"].astype(float)

    res = pd.pivot_table(combined_df, index=["FECHA", "NOMBRESUBDIRECCION", "NOMBREGRUPOESTADISTICO3"], columns= "MODEL", values= "Y_PREDICCION", aggfunc= "mean" ).reset_index()
    res = res.merge(df3, on = ["FECHA", "NOMBRESUBDIRECCION", "NOMBREGRUPOESTADISTICO3"], how = "outer", validate = "1:1")
    res = res.rename(columns = {"FECHA":"MesAnio", "NOMBREGRUPOESTADISTICO3": "NombreGrupoEstadistico3", "NOMBRESUBDIRECCION": "NombreSubdireccion", "Y_HIST":"Y_HIST2"})
    return res


def load_fact_pvo_pv():
    print("Consultando PV")
    pv = data_bq(QUERY_HISTORICO_PV)
    pv["id"] = pv[[ "NombreGrupo", "NombreDireccion", "NombreSubdireccion", "NombreGrupoEstadistico1", "NombreGrupoEstadistico2", "NombreGrupoEstadistico3"]].astype(str).agg("_".join, axis=1)

    print("Consultando PVO")
    pvo = data_bq(QUERY_HISTORICO_PVO)
    pvo["id"] = pvo[[ "NombreGrupo", "NombreDireccion", "NombreSubdireccion", "NombreGrupoEstadistico1", "NombreGrupoEstadistico2", "NombreGrupoEstadistico3"]].astype(str).agg("_".join, axis=1)

    print("Consultando FACT")
    fact = data_bq(QUERY_HISTORICO_FACTURACION)
    fact["id"] = fact[[ "NombreGrupo", "NombreDireccion", "NombreSubdireccion", "NombreGrupoEstadistico1", "NombreGrupoEstadistico2", "NombreGrupoEstadistico3"]].astype(str).agg("_".join, axis=1)
    print(fact.id.nunique())

    out = pvo.merge(pv, how = "outer", on = ["MesAnio","id","NombreGrupo","NombreDireccion","NombreSubdireccion","NombreGrupoEstadistico1","NombreGrupoEstadistico2","NombreGrupoEstadistico3"], validate = "1:1")
    print(out.id.nunique())

    out = out.merge(fact, how = "outer", on = ["MesAnio","id","NombreGrupo","NombreDireccion","NombreSubdireccion","NombreGrupoEstadistico1","NombreGrupoEstadistico2","NombreGrupoEstadistico3"], validate = "1:1")
    print(out.id.nunique())

    for i in ["toneladas_pvo","toneladas_plan_ventas","toneladas_facturadas"]:
        # imputamos nulos con np.nan
        out[i] = out[i].astype(float).fillna(np.nan)

        #limitamos el valor absoluto cercano a cero a cero
        out[i] = np.where(out[i].abs()<0.000001,0, out[i] )

    out["MesAnio"] = pd.to_datetime(out["MesAnio"])

    # SELECICONAMOS G32 DEL
    g32del =  [' 8X19-26 AA GALV QUERETARO', '1X19-36 NEG QUERETARO','ACCESORIOS CROSBY QUERETARO', 'AL GALV CD', 'AL FORJAS C/TRAT (CHQ)',
                'AL FORJAS C/TRAT', 'A.GALV. FINO', 'ALAMBRE PARA GAVIONES','ALAMBRE ESTAÑADO', 'SAW CUT', 'ALAMBRE PULIDO 25-29', 'REC IND (CHQ)',
                'REFACCIONES GRAPADORAS', 'PROTECTOR DE VENTANA', 'CASTILLO DOB.','CABLE DE REFUERZO', 'REC IND FINO', 'CLAVO PARA HERRAR',
                '6X19-26 AF NEG HOUSTON', 'PACKAGED', 'A. GALV. A.C. P/ CABLE', 'TRENZA PARA RESORTE', 'VALLA DEACERO', '8X31-41 AA NEG QUERETARO',
                'INDUSTRIAL TOOLS', 'JOIST HANGER NAILS', 'DERECHO DE VIA','CUT STOCK REBAR', 'GRAPA TAPICERA', 'HEX-NETTING INDUSTRIAL'
                ]
    #  AGREGAMOS ALGUNOS FILTROS...
    out = out[~out.NombreDireccion.isin(['ABAST. EXPORT', 'ABASTECIMIENTOS', 'DAL Y TRANSPORTE GRUPO DEACERO','IIDEA NACIONAL', 'H&H IRON AND METAL INC.', 'MAQUILAS'])]
    out = out[~out.NombreSubdireccion.isin(['EXCEDENTES ACEROS', 'EXCEDENTES ALAMBRES', 'DSS NO METALICOS ', 'DEACERO-MID CONTINENT', 'FILIALES CANADA', 'RESTO ALAMBRES', 'RESTO INGETEK'])]
    out = out[~out.NombreGrupoEstadistico1.isin(['CHATARRA', 'LOGÍSTICA', 'SEGUNDAS', 'INTERNAS PRODUCCION ', "DEACERO POWER"])]
    out = out[~out.NombreGrupoEstadistico2.isin(['2D&3D', 'ALAMBRON', 'ALAMBRON OTROS', 'AMARRADORA DE VARILLA', 'SEGUNDAS / EXCEDENTES', 'PILOTES', 'SERVICIOS INGETEK', 'VARILLA FERRETERA'])]
    out = out[~out.NombreGrupoEstadistico3.isin(g32del)]

    return out

def duplicados(FACT):
    #ELIMINAMOS DUPLICADOS DE FILIALES....
    FACT["id2"] = FACT[["NombreSubdireccion", "NombreGrupoEstadistico3"]].astype(str).agg("_".join, axis=1)
    to_compare = FACT.groupby("id2")["NombreDireccion"].nunique()
    vals_duplicated = to_compare [to_compare>1].index
    #x = FACT[FACT["id2"].isin(vals_duplicated) & (FACT.MesAnio == "2025-08-01")].groupby(["id2","NombreGrupo", "NombreDireccion"]).agg({"toneladas_facturadas":"sum","toneladas_pvo" :"sum"})#.to_csv("file_csv.csv")
    FACT = FACT[~(FACT.id2.isin(vals_duplicated) & (FACT.NombreDireccion == "FILIALES"))]
    return FACT


def calcular_tp_meses(df, lags_years):
    #funcioon que promedia mismos meses de los ultimos n años...
    # Crear los lags usando .transform()
    df["toneladas_facturadas2"] =df["toneladas_facturadas"].fillna(0) 
    
    # Crear los lags correctamente
    for lag in range(1, lags_years + 1):
        df[f'lag_{lag}'] = df.groupby("id")['toneladas_facturadas'].transform(lambda x: x.shift(lag * 12))

    # Calcular TP_MESES como el promedio de los lags
    lag_cols = [f'lag_{lag}' for lag in range(1, lags_years + 1)]
    df['TP_MESES'] = df[lag_cols].mean(axis=1)
    df = df.drop(columns =lag_cols )

    return df.drop(columns = "toneladas_facturadas2")


def calcular_tp_movil(FACT, date_today):
    # funci que calcula media movil para train y teest 
    LIST_DIR = [] 
    grouped = FACT.groupby('id')
    for a_, data in grouped:
        #print(a_, len(LIST_DIR))
        data = data[["MesAnio","toneladas_facturadas"]].set_index("MesAnio")
        data["id"] = a_
        data["TP_MOVIL"] = None
        data["fcst"] = data.toneladas_facturadas
        for date in data.index:
            start_date = date - pd.DateOffset(months=7)
            end_date = date - pd.DateOffset(months=1)
            if date < date_today: 
                data.loc[date, "TP_MOVIL"] = data.loc[(data.index> start_date) & (data.index<= end_date) ,"toneladas_facturadas"].mean()
            else:
                data.loc[date, "fcst"] =  data.loc[(data.index > start_date) & (data.index<= end_date) ,"fcst"].mean()
            data["TP_MOVIL"] = np.where(data.index <date_today, data["TP_MOVIL"], data["fcst"] )
        LIST_DIR.append(data[["id", "TP_MOVIL"]])
    return pd.concat(LIST_DIR).reset_index()
    

def add_zeros(df ,date_today):
     
    print(df.shape)
    fechas = df['MesAnio'].unique()
    id = df['id'].unique()

    # Generar todas las combinaciones posibles
    combinaciones = pd.DataFrame(list(itertools.product(fechas, id)), columns=['MesAnio',"id"])

    for i, col  in enumerate([ "NombreGrupo", "NombreDireccion", "NombreSubdireccion", "NombreGrupoEstadistico1", "NombreGrupoEstadistico2", "NombreGrupoEstadistico3"]):
        # agregamos columnas en el df aplicando el split
        print(i, col)
        combinaciones[col] = combinaciones["id"].str.split("_").apply(lambda x:x[i])

    # Unir con el DataFrame original
    df_completo = pd.merge(combinaciones, df, on = ["MesAnio","id","NombreGrupo","NombreDireccion","NombreSubdireccion","NombreGrupoEstadistico1","NombreGrupoEstadistico2","NombreGrupoEstadistico3"], how='left')
    
    df_completo["valid_ton"] = df_completo[["toneladas_pvo","toneladas_plan_ventas","toneladas_facturadas"]].fillna(0).sum(axis = 1)

    df_completo = df_completo.sort_values(by=['id', 'MesAnio'])
    df["toneladas_facturadas"] =df["toneladas_facturadas"].fillna(0)

    # TP MESES -----------------------------------------------------------------------------------------------
    df_completo = calcular_tp_meses(df_completo, lags_years = 3)
    #df_completo = df_completo.set_index("MesAnio", drop=True)

    # TP MOVIL -----------------------------------------------------------------------------------------------
    TPMOVIL = calcular_tp_movil(df_completo, date_today)
    df_completo = df_completo.merge(TPMOVIL, how = "left", on = ["id", "MesAnio"], validate= "1:1")

    #filtramos ids conflictivos
    df_completo["id2"] = df_completo[["NombreSubdireccion", "NombreGrupoEstadistico3"]].astype(str).agg("_".join, axis=1)
    x = df_completo[(df_completo.NombreGrupo == "TRASPASOS E INTEREMPRESAS") & (df_completo.NombreDireccion == "FILIALES")].groupby("id2").agg({"toneladas_facturadas":"sum","toneladas_pvo" :"sum"})
    ids2del =  x[x["toneladas_pvo"] == 0].index
    df_completo = df_completo[~df_completo.isin(ids2del)]

    df_completo = duplicados(df_completo)
    #FILTRAMOS SOLO 2025 EN ADELANTE    ------------------------------
    df_completo = df_completo[df_completo["MesAnio"].dt.year >=2024 ]
    print(df_completo.id2.nunique())
    return df_completo

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def data_bq(query):
    client = bigquery.Client(project="demanda-prj-dev")
    query = client.query(query)
    pv = query.to_dataframe() #create_bqstorage_client=True
    return pv

def import_validation_data_from_bigquery ():

    mape_midas = data_bq(query_mape_midas_validacion)
    mape_pvo_pv = data_bq(query_mape_pvo_pv_validacion)
    mape_temporalidades_meses = data_bq(query_temporalidades_meses)
    mape_temporalidades_movil = data_bq(query_temporalidades_movil)

    mape_pvo_pv['MAPE_PVO'] = mape_pvo_pv['MAPE_PVO'].astype(float)
    mape_pvo_pv['MAPE_PV'] = mape_pvo_pv['MAPE_PV'].astype(float)
    mape_temporalidades_meses['MAPE_TEMPORALIDADES_MESES'] = mape_temporalidades_meses['MAPE_TEMPORALIDADES_MESES'].astype(float)
    mape_temporalidades_movil['MAPE_TEMPORALIDADES_MOVIL'] = mape_temporalidades_movil['MAPE_TEMPORALIDADES_MOVIL'].astype(float)

    return mape_midas,mape_pvo_pv,mape_temporalidades_meses,mape_temporalidades_movil

def import_forecast_data_from_bigquery ():

    pvo_pv_pronostico = data_bq(query_pvo_pv_pronostico)
    midas_pronostico = data_bq(query_midas_pronostico)

    return pvo_pv_pronostico,midas_pronostico

def MAPE_models (midas,pvo_pv,tp_meses,tp_movil):

    mape_midas = midas[['NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO3','MODEL','MAPE_DEACERO']].copy().drop_duplicates()
    mape_pvo_pv = pvo_pv[['NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO3','MAPE_PVO','MAPE_PV']].copy().drop_duplicates()
    mape_pvo_pv = mape_pvo_pv.groupby(by=['NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO3']).agg({'MAPE_PVO':'mean','MAPE_PV':'mean'}).reset_index()
    mape_temporalidades_meses = tp_meses[tp_meses['MES_TIPO']=='VALIDACION'][['NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO3','MAPE_TEMPORALIDADES_MESES']].copy().drop_duplicates()
    mape_temporalidades_meses = mape_temporalidades_meses.groupby(by=['NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO3']).agg({'MAPE_TEMPORALIDADES_MESES':'mean'}).reset_index()
    mape_temporalidades_movil = tp_movil[tp_movil['MES_TIPO']=='VALIDACION'][['NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO3','MAPE_TEMPORALIDADES_MOVIL']].copy().drop_duplicates()
    mape_temporalidades_movil = mape_temporalidades_movil.groupby(by=['NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO3']).agg({'MAPE_TEMPORALIDADES_MOVIL':'mean'}).reset_index()

    return mape_midas,mape_pvo_pv,mape_temporalidades_meses,mape_temporalidades_movil

def accuracy_analysis (dataframe_analysis):

    dataframe = dataframe_analysis.copy()

    MAPE_fields = ['MAPE_DEACERO','MAPE_PVO','MAPE_PV','MAPE_TEMPORALIDADES_MESES','MAPE_TEMPORALIDADES_MOVIL']#,'MAPE_DM'

    dataframe['Min_MAPE'] = dataframe[MAPE_fields].min(axis = 1)
    mask_midas = dataframe['MAPE_DEACERO'] == dataframe['Min_MAPE']
    mask_pvo = (~mask_midas) & (dataframe['MAPE_PVO'] == dataframe['Min_MAPE'])
    mask_pv = (~mask_midas) & (~mask_pvo) & (dataframe['MAPE_PV'] == dataframe['Min_MAPE'])
    mask_tp = (~mask_midas) & (~mask_pvo) & (~mask_pv) & (dataframe['MAPE_TEMPORALIDADES_MESES'] == dataframe['Min_MAPE'])
    mask_tp_meses = (~mask_midas) & (~mask_pvo) & (~mask_pv) & (~mask_tp) & (dataframe['MAPE_TEMPORALIDADES_MOVIL'] == dataframe['Min_MAPE'])
    #mask_dm = (~mask_midas) & (~mask_pvo) & (~mask_pv) & (~mask_tp) & (~mask_tp_meses) & (dataframe['MAPE_DM'] == dataframe['Min_MAPE'])
    dataframe.loc[mask_pvo, 'MODEL'] = 'PVO'
    dataframe.loc[mask_pv, 'MODEL'] = 'PV'
    dataframe.loc[mask_tp,'MODEL'] = 'TP_MESES'
    dataframe.loc[mask_tp_meses,'MODEL'] = 'TP_MOVIL'
    #dataframe.loc[mask_dm,'MODEL'] = 'DM'

    return dataframe

def evaluation_forecast_integration (dataframe_analysis,midas_forecast,midas_evaluation,pvo_pv_forecast,pvo_pv_evaluation,tp_meses,tp_movil,dm_forecast,dm_evaluation):

    lista_evaluacion = list()
    lista_pronosticos = list()

    for idx in dataframe_analysis.index:

        granularidad = dataframe_analysis.iloc[idx,0]
        granularidad_profunda = dataframe_analysis.iloc[idx,1]
        modelo = dataframe_analysis.iloc[idx,2]

        if modelo in ['PVO','PV']:

            forecast_business = pvo_pv_forecast[(pvo_pv_forecast[GRANULARITY]==granularidad)&(pvo_pv_forecast[DEEPGRANULARITY]==granularidad_profunda)].copy()
            forecast_business = forecast_business[['FECHA',GRANULARITY,DEEPGRANULARITY,modelo]].rename(columns={modelo:'PRONOSTICO'})
            forecast_business.insert(3,'MODEL',modelo)
            lista_pronosticos.append( forecast_business)

            evaluation_business = pvo_pv_evaluation[(pvo_pv_evaluation[GRANULARITY]==granularidad)&(pvo_pv_evaluation[DEEPGRANULARITY]==granularidad_profunda)].copy()
            evaluation_business = evaluation_business[['FECHA',GRANULARITY,DEEPGRANULARITY,f'MAPE_{modelo}']].rename(columns={f'MAPE_{modelo}':'MAPE'})
            evaluation_business.insert(3,'MODEL',modelo)
            lista_evaluacion.append( evaluation_business)

        elif modelo == 'TP_MESES':

            tp_forecast_business = tp_meses[(tp_meses['MES_TIPO']=='PRONOSTICO')&(tp_meses[GRANULARITY]==granularidad)&(tp_meses[DEEPGRANULARITY]==granularidad_profunda)].copy()
            tp_forecast_business = tp_forecast_business[['FECHA',GRANULARITY,DEEPGRANULARITY,modelo]].rename(columns={modelo:'PRONOSTICO'})
            tp_forecast_business.insert(3,'MODEL',modelo)
            lista_pronosticos.append( tp_forecast_business)


            tp_evaluation_business = tp_meses[(tp_meses['MES_TIPO']=='VALIDACION')&(tp_meses[GRANULARITY]==granularidad)&(tp_meses[DEEPGRANULARITY]==granularidad_profunda)].copy()
            tp_evaluation_business = tp_evaluation_business[['FECHA',GRANULARITY,DEEPGRANULARITY,'MAPE_TEMPORALIDADES_MESES']].rename(columns={'MAPE_TEMPORALIDADES_MESES':'MAPE'})
            tp_evaluation_business.insert(3,'MODEL',modelo)
            lista_evaluacion.append(tp_evaluation_business)

        elif modelo == 'TP_MOVIL':

            tp_meses_forecast_business = tp_movil[(tp_movil['MES_TIPO']=='PRONOSTICO')&(tp_movil[GRANULARITY]==granularidad)&(tp_movil[DEEPGRANULARITY]==granularidad_profunda)].copy()
            tp_meses_forecast_business = tp_meses_forecast_business[['FECHA',GRANULARITY,DEEPGRANULARITY,modelo]].rename(columns={modelo:'PRONOSTICO'})
            tp_meses_forecast_business.insert(3,'MODEL',modelo)
            lista_pronosticos.append( tp_meses_forecast_business)

            tp_meses_evaluation_business = tp_movil[(tp_movil['MES_TIPO']=='VALIDACION')&(tp_movil[GRANULARITY]==granularidad)&(tp_movil[DEEPGRANULARITY]==granularidad_profunda)].copy()
            tp_meses_evaluation_business = tp_meses_evaluation_business[['FECHA',GRANULARITY,DEEPGRANULARITY,'MAPE_TEMPORALIDADES_MOVIL']].rename(columns={'MAPE_TEMPORALIDADES_MOVIL':'MAPE'})
            tp_meses_evaluation_business.insert(3,'MODEL',modelo)
            lista_evaluacion.append(tp_meses_evaluation_business)

        elif modelo == 'DM':

            dm_forecast_business = dm_forecast[(dm_forecast[GRANULARITY]==granularidad)&(dm_forecast[DEEPGRANULARITY]==granularidad_profunda)].copy()
            dm_forecast_business = dm_forecast_business[['FECHA',GRANULARITY,DEEPGRANULARITY,modelo]].rename(columns={modelo:'PRONOSTICO'})
            dm_forecast_business.insert(3,'MODEL',modelo)
            lista_pronosticos.append( dm_forecast_business )

            dm_evaluation_business = dm_evaluation[(dm_evaluation[GRANULARITY]==granularidad)&(dm_evaluation[DEEPGRANULARITY]==granularidad_profunda)].copy()
            dm_evaluation_business = dm_evaluation_business[['FECHA',GRANULARITY,DEEPGRANULARITY,'MAPE_DM']].rename(columns={'MAPE_DM':'MAPE'})
            dm_evaluation_business.insert(3,'MODEL',modelo)
            lista_evaluacion.append(dm_evaluation_business)

        else:

            forecast_midas = midas_forecast[(midas_forecast[GRANULARITY]==granularidad)&(midas_forecast[DEEPGRANULARITY]==granularidad_profunda)&(midas_forecast['MODEL']==modelo)].copy()
            forecast_midas.rename(columns={'MIDAS':'PRONOSTICO'},inplace=True)
            lista_pronosticos.append( forecast_midas )

            evaluation_midas = midas_evaluation[(midas_evaluation[GRANULARITY]==granularidad)&(midas_evaluation[DEEPGRANULARITY]==granularidad_profunda)&(midas_evaluation['MODEL']==modelo)].copy()
            evaluation_midas.rename(columns={'MAPE_DEACERO':'MAPE'},inplace=True)
            lista_evaluacion.append( evaluation_midas )

    dataframe_pronosticos = pd.concat(lista_pronosticos)
    dataframe_pronosticos = pd.merge(dataframe_pronosticos,data_bq(query_historico_facturacion),
                                     left_on=[GRANULARITY,DEEPGRANULARITY],
                                     right_on=[GRANULARITY_HISTORICO,DEEPGRANULARITY_HISTORICO],
                                     how='left')
    dataframe_pronosticos['FECHA'] = pd.to_datetime(dataframe_pronosticos['FECHA'])

    dataframe_evaluacion = pd.concat(lista_evaluacion)

    evaluacion_historico = data_bq(query_historico_facturacion_evaluacion)
    midas_historico = data_bq(query_midas_pronostico_evaluacion)
    pvo_pvo_historico = data_bq(query_mape_pvo_pvo_pronostico_evaluacion)
    temporalidades_meses_historico = data_bq(query_temporalidades_meses_pronostico_evaluacion)
    temporalidades_movil_historico = data_bq(query_temporalidades_movil)
    temporalidades_movil_historico = temporalidades_movil_historico[temporalidades_movil_historico['MES_TIPO']=='VALIDACION'].reset_index(drop=True)
    temporalidades_movil_historico['MODEL'] = 'TP_MOVIL'

    midas_historico['FECHA'] = pd.to_datetime( midas_historico['FECHA'] )
    dataframe_evaluacion['FECHA'] = pd.to_datetime( dataframe_evaluacion['FECHA'] )
    evaluacion_historico['FECHA'] = pd.to_datetime( evaluacion_historico['FECHA'] )
    pvo_pvo_historico['FECHA'] = pd.to_datetime( pvo_pvo_historico['FECHA'] )
    temporalidades_meses_historico['FECHA'] = pd.to_datetime( temporalidades_meses_historico['FECHA'] )
    temporalidades_movil_historico['FECHA'] = pd.to_datetime( temporalidades_movil_historico['FECHA'] )

    dataframe_evaluacion = pd.merge(dataframe_evaluacion,evaluacion_historico,
                                    on=['FECHA',GRANULARITY,DEEPGRANULARITY],
                                    how='left'
                                    )
    dataframe_evaluacion = pd.merge(dataframe_evaluacion,midas_historico,
                                    on=['FECHA',GRANULARITY,DEEPGRANULARITY,'MODEL'],
                                    how='left'
                                    )
    dataframe_evaluacion = pd.merge(dataframe_evaluacion,pvo_pvo_historico.rename(columns={'MODELO_PVO':'MODEL'})[['FECHA',GRANULARITY,DEEPGRANULARITY,'MODEL','PREDICCION_PVO']],
                                    on=['FECHA',GRANULARITY,DEEPGRANULARITY,'MODEL'],
                                    how='left'
                                    )
    dataframe_evaluacion = pd.merge(dataframe_evaluacion,pvo_pvo_historico.rename(columns={'MODELO_PV':'MODEL'})[['FECHA',GRANULARITY,DEEPGRANULARITY,'MODEL','PREDICCION_PV']],
                                on=['FECHA',GRANULARITY,DEEPGRANULARITY,'MODEL'],
                                how='left'
                                )
    dataframe_evaluacion = pd.merge(dataframe_evaluacion,temporalidades_meses_historico.rename(columns={'MODELO_TP_MESES':'MODEL'})[['FECHA',GRANULARITY,DEEPGRANULARITY,'MODEL','PREDICCION_TP_MESES']],
                                on=['FECHA',GRANULARITY,DEEPGRANULARITY,'MODEL'],
                                how='left'
                                )
    dataframe_evaluacion = pd.merge(dataframe_evaluacion,temporalidades_movil_historico.rename(columns={'TP_MOVIL':'PREDICCION_TP_MOVIL'})[['FECHA',GRANULARITY,DEEPGRANULARITY,'MODEL','PREDICCION_TP_MOVIL']],
                                on=['FECHA',GRANULARITY,DEEPGRANULARITY,'MODEL'],
                                how='left'
                                )
    
    dataframe_evaluacion['PRONOSTICO'] = dataframe_evaluacion[['PREDICCION_MIDAS', 'PREDICCION_PVO', 'PREDICCION_PV', 'PREDICCION_TP_MESES', 'PREDICCION_TP_MOVIL']].sum(axis=1, skipna=True)

    dataframe_evaluacion.drop(columns=['PREDICCION_MIDAS', 'PREDICCION_PVO', 'PREDICCION_PV', 'PREDICCION_TP_MESES', 'PREDICCION_TP_MOVIL'],inplace=True)

    if DEEPGRANULARITY == 'NOMBREGRUPOESTADISTICO3':
        dataframe_pronosticos.drop(columns=[DEEPGRANULARITY_HISTORICO,'NombreSubdireccion'],inplace=True)

    dataframe_pronosticos.rename(columns={'NombreGrupo':'NOMBREGRUPO','NombreDireccion':'NOMBREDIRECCION','NombreSubdireccion':'NOMBRESUBDIRECCION',
                                        'NombreGrupoEstadistico1':'NOMBREGRUPOESTADISTICO1','NombreGrupoEstadistico2':'NOMBREGRUPOESTADISTICO2'},inplace=True)

    dataframe_pronosticos = dataframe_pronosticos[ordenamiento_columnas]

    timezone_cdmx = pytz.timezone('America/Mexico_City')
    fecha_ejecucion = datetime.datetime.now(timezone_cdmx).strftime('%B-%Y')

    dataframe_pronosticos.insert(0,'FECHA_EJECUCION',fecha_ejecucion)
    dataframe_pronosticos.dropna(inplace=True)
    dataframe_pronosticos['PRONOSTICO'] = dataframe_pronosticos['PRONOSTICO'].astype(float)

    return dataframe_pronosticos,dataframe_evaluacion

def automatizacion_comparativa_resultados ():

    # Extraemos la informacion de BigQuery de MIDAS, PVO, PV, TP MESES y TP MOVIL
    midas_evaluation,pvo_pv_evaluation,temporalidades_meses,temporalidades_movil = import_validation_data_from_bigquery()

    # Extraemos la informacion de pronostico de MIDAS, PVO y PV
    pvo_pv_forecast,midas_forecast = import_forecast_data_from_bigquery()

    # Extraemos la informacion de facturacion historica
    historico_facturacion = data_bq(query_historico_facturacion)

    #mape_dm = DM_validacion()
    #dm_pronostico = DM_pronostico()
    #mape_dm['MAPE_DM'] = mape_dm['MAPE_DM'].astype(float)

    # Calculamos el MAPE de los modelos por combinacion
    mape_midas,mape_pvo_pv,mape_temporalidades_meses,mape_temporalidades_movil = MAPE_models(midas=midas_evaluation,
                                                                                             pvo_pv=pvo_pv_evaluation,
                                                                                             tp_meses=temporalidades_meses,
                                                                                             tp_movil=temporalidades_movil)

    #mape_dm_unique = mape_dm[['NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO3','MAPE_DM']].copy().drop_duplicates()
    #mape_dm_unique = mape_dm_unique.groupby(by=['NOMBRESUBDIRECCION','NOMBREGRUPOESTADISTICO3']).agg({'MAPE_DM':'mean'}).reset_index()

    # Unimos los datos
    dataframe_0 = pd.merge(mape_midas,mape_pvo_pv,on=[GRANULARITY, DEEPGRANULARITY],how='left')
    dataframe_1 = pd.merge(dataframe_0,mape_temporalidades_meses,on=[GRANULARITY, DEEPGRANULARITY],how='left')
    dataframe = pd.merge(dataframe_1,mape_temporalidades_movil,on=[GRANULARITY, DEEPGRANULARITY],how='left')
    
    #dataframe = pd.merge(
    #    dataframe_2,
    #    mape_dm_unique,
    #    on=[GRANULARITY, DEEPGRANULARITY],
   #     how='left'
    #)

    # Analisis de asertividad

    dataframe = accuracy_analysis(dataframe)

    # Algoritmo de integracion de pronosticos y evaluacion
            
    dataframe_pronosticos,dataframe_evaluacion = evaluation_forecast_integration(dataframe_analysis=dataframe,
                                                                                 midas_forecast=midas_forecast,
                                                                                 midas_evaluation=midas_evaluation,
                                                                                 pvo_pv_forecast=pvo_pv_forecast,
                                                                                 pvo_pv_evaluation=pvo_pv_evaluation,
                                                                                 tp_meses=temporalidades_meses,
                                                                                 tp_movil=temporalidades_movil,
                                                                                 dm_forecast=None,
                                                                                 dm_evaluation=None)

    dataframe_pronostico = pd.merge(dataframe_pronosticos,dataframe,on=[GRANULARITY,DEEPGRANULARITY],how='left')
    dataframe_pronostico.drop(columns=['MODEL_y','MAPE_DEACERO','MAPE_PVO','MAPE_PV','MAPE_TEMPORALIDADES_MESES','MAPE_TEMPORALIDADES_MOVIL',#'MAPE_DM'
                                       ],inplace=True)
    dataframe_pronostico.rename(columns={'MODEL_x':'MODEL','Min_MAPE':'MIN_MAPE'},inplace=True)
    dataframe.rename(columns={'MAPE_DEACERO':'MAPE_MIDAS'},inplace=True)
    dataframe_mape = dataframe.copy()

    #dataframe_pronostico.to_csv("C:/Users/JLOPCRU/OneDrive - deacero.com/Escritorio/Proyectos/Demanda/Automatizacion comparativa resultados MIDAS/pronosticos.csv",index=False)
    #dataframe_evaluacion.to_csv("C:/Users/JLOPCRU/OneDrive - deacero.com/Escritorio/Proyectos/Demanda/Automatizacion comparativa resultados MIDAS/evaluacion.csv",index=False)
    #dataframe.to_csv("C:/Users/JLOPCRU/OneDrive - deacero.com/Escritorio/Proyectos/Demanda/Automatizacion comparativa resultados MIDAS/mape.csv",index=False)
    
    return dataframe_mape.reset_index(drop=True),dataframe_evaluacion.reset_index(drop=True),dataframe_pronostico.reset_index(drop=True)

def tabla_mape(dataframe):
    dataframe_mape = dataframe.copy()
    dataframe_mape = dataframe_mape['MODEL'].value_counts().reset_index()
    dataframe_mape.rename(columns={'count':'COMBINACIONES'},inplace=True)
    dataframe_mape.loc[~dataframe_mape['MODEL'].isin(['PVO','PV','TP_MESES','TP_MOVIL','DM']),'MODEL'] = 'MIDAS'
    dataframe_mape = dataframe_mape.groupby(by=['MODEL']).agg({'COMBINACIONES':'sum'}).reset_index()
    dataframe_mape['PROPORCION'] = (dataframe_mape['COMBINACIONES'] / dataframe_mape['COMBINACIONES'].sum())*100
    dataframe_mape['PROPORCION'] = dataframe_mape['PROPORCION'].apply(lambda x: f"{x:.1f} %")
    return dataframe_mape.sort_values(by='COMBINACIONES',ascending=False).reset_index(drop=True)
