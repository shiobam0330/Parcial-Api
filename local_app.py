import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from pycaret.classification import *

path = "C:/Users/MCS/OneDrive - Universidad Santo Tomás/Inteligencia Artificial/Codigos Propios/PARCIAL 1 CORTE/"
#path = "D:/Downloads/PARCIAL IA/"
train = pd.read_csv(path + "train.csv")
train = train.drop(columns = ["id"])
train = train.drop(columns = ["Target"])
train.rename(columns = {"Mother's occupation": "Mother occupation",
                            "Father's occupation": "Father occupation",
                            "Mother's qualification": "Mother qualification",
                            "Father's qualification": "Father qualification"}, inplace = True)

categoricas1  = ['Marital status', 'Application mode', 'Application order', 'Course',
       'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
       'Mother qualification', 'Father qualification', 'Mother occupation',
       'Father occupation', 'Displaced', 'Educational special needs', 'Debtor',
       'Tuition fees up to date', 'Gender', 'Scholarship holder',
       'International']

for k in categoricas1:
    train[k] = train[k].astype("O")

with open(path + 'best_model.pkl', 'rb') as file:
    modelo = pickle.load(file)
def num(prev_qua_grade, adm_grade, age, cur_1c, cur_1en, cur_1ev, cur_1a, cur_1g, cur_1w, cur_2c):
    numericas = pd.DataFrame({'Previous qualification (grade)':[float(prev_qua_grade)], 
                              'Admission grade':[float(adm_grade)],
                              'Age at enrollment':[float(age)], 
                              'Curricular units 1st sem (credited)':[float(cur_1c)], 
                              'Curricular units 1st sem (enrolled)':[float(cur_1en)],
                              'Curricular units 1st sem (evaluations)':[float(cur_1ev)],
                              'Curricular units 1st sem (approved)':[float(cur_1a)],
                              'Curricular units 1st sem (grade)':[float(cur_1g)],
                              'Curricular units 1st sem (without evaluations)':[float(cur_1w)],
                              'Curricular units 2nd sem (credited)':[float(cur_2c)],
                              'Curricular units 2nd sem (enrolled)':[float(cur_2en)],
                              'Curricular units 2nd sem (evaluations)':[float(cur_2ev)],
                              'Curricular units 2nd sem (approved)':[float(cur_2a)],
                              'Curricular units 2nd sem (grade)':[float(cur_2g)],
                              'Curricular units 2nd sem (without evaluations)':[float(cur_2w)], 
                              'Unemployment rate':[float(unem)],
                              'Inflation rate':[float(inflation)], 
                              'GDP':[float(gdp)]})
    return numericas
def categ(m_status, app_mode, app_order, course, day_time, prev_qua, nac, mother_qua, father_qua, 
          mother_occup, father_occup, displaced, edu_spec, debtor, tuition, gender, scholar, intern):
     categoricas = pd.DataFrame({'Marital status':[m_status], 
                                 'Application mode':[app_mode], 
                                 'Application order':[app_order], 
                                 'Course':[course],
                                 'Daytime/evening attendance':[day_time], 
                                 'Previous qualification':[prev_qua], 
                                 'Nacionality':[nac],
                                 'Mother qualification':[mother_qua], 
                                 'Father qualification':[father_qua], 
                                 'Mother occupation':[mother_occup],
                                 'Father occupation':[father_occup], 
                                 'Displaced':[displaced], 
                                 'Educational special needs':[edu_spec],
                                 'Debtor':[debtor],
                                 'Tuition fees up to date':[tuition], 
                                 'Gender':[gender], 
                                 'Scholarship holder':[scholar],
                                 'International':[intern]})
     return categoricas

numericas1 = ['Previous qualification (grade)', 'Admission grade',
       'Age at enrollment', 'Curricular units 1st sem (credited)',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)',
       'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
       'Inflation rate', 'GDP']
def nombre_(x):
  return "C"+str(x)

c2 = ['Curricular units 2nd sem (approved)_2',
 'Curricular units 2nd sem (enrolled)_2',
 'Curricular units 2nd sem (evaluations)_2']
cxy = ['Curricular units 2nd sem (approved)__Curricular units 2nd sem (grade)',
 'Admission grade__Curricular units 2nd sem (approved)',
 'Curricular units 1st sem (approved)__Curricular units 2nd sem (approved)']
razxy= ['Curricular units 2nd sem (approved)__coc__Curricular units 1st sem (enrolled)',
 'Curricular units 2nd sem (approved)__coc__Curricular units 2nd sem (enrolled)',
 'Curricular units 2nd sem (approved)__coc__Age at enrollment']
catxy = ['Tuition fees up to date__Scholarship holder_C1__C1',
 'Educational special needs__Tuition fees up to date_C0__C1',
 'Gender__Scholarship holder_C1__C0']
cuactxy = ['Curricular units 2nd sem (approved)_Tuition fees up to date_1',
 'Curricular units 2nd sem (evaluations)_Tuition fees up to date_1',
 'Curricular units 1st sem (approved)_Tuition fees up to date_1']

def ingieneria(train):
	D1 = train.get(numericas1).copy()
	D2 = train.get(categoricas1).copy()
	for k in categoricas:
		D2[k] = D2[k].map(nombre_)
	D4 = D2.copy()
	cuadrado = [re.findall(r'(.+)_\d+', item) for item in c2]
	cuadrado = [x[0] for x in cuadrado]
	for k in cuadrado:
		D1[k+"_2"] = D1[k] ** 2
	result = [re.findall(r'([A-Za-z\s\(\)0-9]+)', item) for item in cxy]
	for k in result:
		D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
	result2 = [re.findall(r'(.+)__coc__(.+)', item) for item in razxy]
	for k in result2:
		k2 = k[0]
		D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
	result3 = [re.search(r'([^_]+__[^_]+)', item).group(1).split('__') for item in catxy]
	for k in result3:
		D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
	D5 = train.copy()
	result4 = [re.search(r'(.+?)_(.+?)_1', item).groups() for item in cuactxy]
	contador = 0
	for k in result4:
		col1, col2 = k[1], k[0] # categórica, cuantitativa
		if contador == 0:
			D51 = pd.get_dummies(D5[col1],drop_first=True)
			for j in D51.columns:
				D51[j] = D51[j] * D5[col2]
			D51.columns = [col2+"_"+col1+"_"+ str(x) for x in D51.columns]
		else:
			D52 = pd.get_dummies(D5[col1],drop_first=True)
			for j in D52.columns:
				D52[j] = D52[j] * D5[col2]
			D52.columns = [col2+"_"+col1+"_"+ str(x) for x in D52.columns]
			D51 = pd.concat([D51,D52],axis=1)
		contador = contador + 1
	B1 = pd.concat([D1,D4],axis=1)
	base_modelo = pd.concat([B1,D51],axis=1)
	return base_modelo

def indicadora2(x):
  if x==0:
    return "Graduate"
  elif x== 1:
    return "Enrolled"
  else:
    return "Dropout"
st.title("Predicción de de la APP basada en características de usuario")

m_status = st.selectbox("Seleccione Marital status:", ['1', '2', '3', '4', "5", "6"]) 
app_mode = st.selectbox("Seleccione Application mode:", ["1","2","3","4","5","7","9","10","12","15",
                                                         "16","17","18","26", "27","35","39","42",
                                                         "43","44","51","53"]) 
app_order = st.selectbox("Seleccione Application order:", ["1", "0", "2", "3","4","5", "6", "9"]) 
course = st.selectbox("Seleccione Course:", ["9238", "9254", "9500", "171", "9085", "9773", "9003", "9853",
                                              "9147", "9670","8014", "9119", "9991", "9130", "9556", "9070", 
                                              "33", "979", "39"])
day_time = st.selectbox("Seleccione Daytime/evening attendance:", ["1", "0"])
prev_qua = st.selectbox("Seleccione Previous qualification:", ["1","2", "3","4","5","6", "9","10","11","12",
                                                                "14", "15", "17", "19","36", "37", "38","39",
                                                                  "40","42","43"])
nac = st.selectbox("Seleccione Nacionality:", ["1", "2", "6", "11", "17", "21", "22", "24", "25", "26", "32",
                                                "41", "62", "100", "101", "103", "105","109"]) 
mother_qua = st.selectbox("Seleccione Mother qualification:", ['1', '2', '6', '11', '17', '21', '22', '24', 
                                                               '25', '26', '32', '41', '62', '100', '101', 
                                                               '103', '105', '109']) 
father_qua = st.selectbox("Seleccione Father qualification:", ['19','1', '2', '3', '4', '5', '6', '7', '9', '10',
                                                                '11', '12', '13', '14', '15', '18', '20',
                                                                '21', '22', '23', '24', '25', '26','27', '29',
                                                                '30', '31', '33', '34', '35', '36', '37', '38',
                                                                '39','40', '41', '42', '43', '44']) 
mother_occup = st.selectbox("Seleccione Mother occupation:", ['5','0', '1', '2', '3', '4', '6', '7', '8', '9', 
                                                              '10', '11', '38', '90', '99', '101', '103', '122', 
                                                              '123', '124', '125', '127', '131', '132', '134', 
                                                              '141', '143', '144', '151', '152','153', '163', 
                                                              '171', '172', '173', '175', '191', '192', '193',
															  '194']) 
father_occup = st.selectbox("Seleccione Father occupation:", ['5', '0', '1', '2', '3', '4', '6', '7', '8', '9', 
                                                              '10', '11', '12','13', '19', '22', '39', '90', '96', 
                                                              '99', '101', '102', '103', '112', '114', '121', 
                                                              '122', '123', '124', '125', '131', '132', '134', 
                                                              '135', '141', '143', '144', '148', '151', '152', 
                                                              '153', '154', '161', '163', '171', '172', '174', 
                                                              '175', '181', '182', '183', '191', '192', '193',
															  '194', '195']) 
displaced = st.selectbox("Seleccione Displaced:", ['0','1']) 
edu_spec = st.selectbox("Seleccione Educational special needs:", ['0','1']) 
debtor = st.selectbox("Seleccione Debtor:",['0','1']) 
tuition = st.selectbox("Seleccione Tuition fees up to date:", ['1','0']) 
gender = st.selectbox("Seleccione Gender:", ['0','1']) 
scholar = st.selectbox("Scholarship holder:", ['1','0']) 
intern = st.selectbox("International:", ['0','1']) 

prev_qua_grade = st.text_input("Ingrese Previous qualification (grade):", value="126")
adm_grade = st.text_input("Ingrese Admission grade:", value="122.6")
age = st.text_input("Ingrese Age at enrollment:", value="18")
cur_1c = st.text_input("Ingrese Curricular units 1st sem (credited):", value="0")
cur_1en = st.text_input("Ingrese Curricular units 1st sem (enrolled):", value="6")
cur_1ev = st.text_input("Ingrese Curricular units 1st sem (evaluations):", value="6")
cur_1a = st.text_input("Ingrese Curricular units 1st sem (approved):", value="6")
cur_1g = st.text_input("Ingrese Curricular units 1st sem (grade):", value="14.5")
cur_1w = st.text_input("Ingrese Curricular units 1st sem (without evaluations):", value="0")
cur_2c = st.text_input("Ingrese Curricular units 2nd sem (credited):", value="0")
cur_2en = st.text_input("Ingrese Curricular units 2nd sem (enrolled):", value="6")
cur_2ev = st.text_input("Ingrese Curricular units 2nd sem (evaluations):", value="7")
cur_2a = st.text_input("Ingrese Curricular units 2nd sem (approved):", value="6")
cur_2g = st.text_input("Ingrese Curricular units 2nd sem (grade):", value="12.428571")
cur_2w = st.text_input("Ingrese Curricular units 2nd sem (without evaluations):", value="0")
unem = st.text_input("Ingrese Unemployment rate:", value="11.1")
inflation = st.text_input("Ingrese Inflation rate:", value="0.6")
gdp = st.text_input("Ingrese GDP:", value="2.02")


if st.button("Calcular"):
	try:
		numericas = num(prev_qua_grade, adm_grade, age, cur_1c, cur_1en, cur_1ev, cur_1a, cur_1g, cur_1w, cur_2c)
		categoricas =categ(m_status, app_mode, app_order, course, day_time, prev_qua, nac, mother_qua, father_qua, 
						   mother_occup, father_occup, displaced, edu_spec, debtor, tuition, gender, scholar, intern)
		nuevos = pd.concat([numericas, categoricas], axis=1)
		train = pd.concat([train, nuevos], axis=0, ignore_index=True)
		cols_numericas =['Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance', 'Previous qualification', 'Nacionality', 
				   'Mother qualification', 'Father qualification', 'Mother occupation',
				   'Father occupation', 'Displaced', 'Educational special needs', 'Debtor',
				   'Tuition fees up to date', 'Gender', 'Scholarship holder','International']
		for col in cols_numericas:
			train[col] = pd.to_numeric(train[col], errors='coerce')
		base_modelo = ingieneria(train)
		prediccion = predict_model(modelo, data = base_modelo)
		prediccion["prediction_label"] = prediccion["prediction_label"].map(indicadora2)
		prediccion = prediccion["prediction_label"].tail(1).values[0]
		st.markdown(f"<p class='big-font'>Predicción: {prediccion}</p>", unsafe_allow_html=True) 
	except ValueError:
		st.error("Por favor, ingrese valores numéricos válidos en todos los campos.")
		
if st.button("Reiniciar"):
    st.experimental_rerun()
