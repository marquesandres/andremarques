"""

Modelo de crédito Fenix.

Autor: André Marques

"""

import os
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import sklearn.metrics as metrics
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz

def imprime_resultados(results):
  media = results['test_score'].mean()
  desvio_padrao = results['test_score'].std()
  print("Accuracy médio: %.2f" % (media * 100))
  print("Accuracy intervalo: [%.2f, %.2f]" % ((media - 2 * desvio_padrao)*100, (media + 2 * desvio_padrao) * 100))

def imprime_score(scores):
  media = scores.mean() * 100
  desvio = scores.std() * 100
  print("Accuracy médio %.2f" % media)
  print("Intervalo [%.2f, %.2f]" % (media - 2 * desvio, media + 2 * desvio))

# mostra o caminho que o atom esta buscando o arquivo

cwd = os.getcwd()
cwd

# Mostrar todas as colunas e linhas do DataFrame

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

dados1 = pd.read_excel('/home/andre/gdrive/DataScience/documents/desenvolvimento/fenix/tratamento/BASEJATAI.xlsx', sheet_name="BASE DE DADOS JATAÍ")

dados2 = pd.read_excel('/home/andre/gdrive/DataScience/documents/desenvolvimento/fenix/tratamento/BaseCliente.xlsx')

base_teste = pd.read_excel('/home/andre/gdrive/DataScience/documents/desenvolvimento/fenix/base_teste/base_test.xlsx')

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

print(dados1.head())


dados2.head()

dados1 = pd.DataFrame(dados1)

dados2 = pd.DataFrame(dados2)

base_teste = pd.DataFrame(base_teste)

dados1 = dados1.dropna(subset=['CPF'])

dados1.head()

dados = pd.merge(dados1, dados1, how='left', on='CPF')

dados = dados1.set_index('CPF').join(dados2.set_index('CPF'))

dados.head()

dados.columns = [
    'empreendimento',
    'titulo',
    'cliente',
    'soma_de_total',
    'dias_de_atraso',
    'classificacao',
    'parecer_credito_fenix',
    'observacao',
    'nome',
    'data_de_nascimento',
    'score',
    'chance_de_pagamento',
    'classificacao_risco',
    'risco_de_inadimplencia',
    'capacidade_mensal_de_pagamento',
    'comprometimento_mensal_estimado',
    'renda_estimada']

dados.classificacao.unique()

map = {'Adimplente': 0, 'Até 30 dias': 1}

dados['classificacao'] = dados['classificacao'].map(map)

dados_com_dummy = pd.get_dummies(
    dados,
    columns=['parecer_credito_fenix', 'classificacao_risco'],
    drop_first=True,
)

dados_com_dummy = dados_com_dummy.dropna()

dados_com_dummy.reset_index(level=0, inplace=True)


# 'dias_de_atraso',

x=dados_com_dummy[[ 'score','chance_de_pagamento', # soma_total é o valor da parcela
                    'risco_de_inadimplencia',
                    'capacidade_mensal_de_pagamento',
                    'comprometimento_mensal_estimado',
                    'renda_estimada'
                    ] ]  

y = dados_com_dummy['classificacao']

x.shape

x.corr()

dados_com_dummy.shape

# Modelo de Arvore GridSearchCV Sem Kflod


SEED=301
np.random.seed(SEED)

espaco_de_parametros = {
    "max_depth" : [5, 10],         #  10,12
    "min_samples_split" : [32, 64, 128],
    "min_samples_leaf" : [32, 64, 128],
    "criterion" : ["gini", "entropy"]
}

busca = GridSearchCV(DecisionTreeClassifier(),
                    espaco_de_parametros,
                    cv = KFold(n_splits = 5, shuffle=True))
busca.fit(x, y)
resultados = pd.DataFrame(busca.cv_results_)
resultados.head()


scores = cross_val_score(busca, x, y, cv = KFold(n_splits=5, shuffle=True))
scores


imprime_score(scores)

melhor = busca.best_estimator_
print(melhor)



# Gráfico da arvore

features = x.columns
dot_data = export_graphviz(melhor, out_file=None, filled=True, rounded=True,
                          class_names=["não","sim"],
                          feature_names = features)
graph = graphviz.Source(dot_data)
graph


# treinando o modelo 

SEED=301
np.random.seed(SEED)

espaco_de_parametros = {
    "max_depth" : [5],         
    "min_samples_split" : [32],
    "min_samples_leaf" : [128],
    "criterion" : ["gini"]
}

modelo = GridSearchCV(DecisionTreeClassifier(),
                    espaco_de_parametros,
                    cv = KFold(n_splits = 5, shuffle=True))
modelo.fit(x, y)


teste=base_teste[[ 'score','chance_de_pagamento', 
                    'risco_de_inadimplencia',
                    'capacidade_mensal_de_pagamento',
                    'comprometimento_mensal_estimado',
                    'renda_estimada'
                    ] ]  

modelo.predict(teste)


