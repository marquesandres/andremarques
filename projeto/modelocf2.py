"""

Modelo de crédito Fenix.

Autor: André Marques

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

# mostra o caminho que o atom esta buscando o arquivo

cwd = os.getcwd()
cwd

# Mostrar todas as colunas e linhas do DataFrame

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

dados1 = pd.read_excel('/home/andre/gdrive/DataScience/documents/desenvolvimento/fenix/tratamento/BASEJATAI.xlsx', sheet_name="BASE DE DADOS JATAÍ")

dados2 = pd.read_excel('/home/andre/gdrive/DataScience/documents/desenvolvimento/fenix/tratamento/BaseCliente.xlsx')

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

print(dados1.head())


dados2.head()

dados1 = pd.DataFrame(dados1)

dados2 = pd.DataFrame(dados2)

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
    drop_first=True)

dados_com_dummy = dados_com_dummy.dropna()



x = dados_com_dummy[[ 'score','chance_de_pagamento', # soma_total é o valor da parcela
                    'risco_de_inadimplencia',
                    'capacidade_mensal_de_pagamento',
                    'comprometimento_mensal_estimado',
                    'renda_estimada'
                    ] ]

y = dados_com_dummy['classificacao']

x.shape

x.corr()

dados_com_dummy.shape

SEED = 77
treino_x, teste_x, treino_y, teste_y = train_test_split(
    x, y, test_size=0.30, random_state=SEED
)

base_treino = treino_x.shape[0]
base_teste = teste_x.shape[0]

print(
    'A base de treino tem %s elementos e a base de teste tem %s elementos.'
    % (base_treino, base_teste)
)


SEED = 77
treino_x, teste_x, treino_y, teste_y = train_test_split(
    x, y, test_size=0.30, random_state=SEED
)

base_treino = treino_x.shape[0]
base_teste = teste_x.shape[0]

print(
    'A base de treino tem %s elementos e a base de teste tem %s elementos.'
    % (base_treino, base_teste)
)


modelo = LogisticRegression(max_iter=1000)
modelo.fit(treino_x, treino_y)
print(modelo.score(treino_x, treino_y))


previsoes = modelo.predict(teste_x)
previsoes


acuracia = accuracy_score(teste_y, previsoes)
acuracia = round(acuracia, 3) * 100
acuracia

matriz_confusao = plot_confusion_matrix(
    modelo, teste_x, teste_y, cmap='Blues', values_format='.3g'
)
matriz_confusao

print(classification_report(teste_y, previsoes))

prob_previsao = modelo.predict_proba(teste_x)[:, 1]

tfp, tvp, limite = roc_curve(teste_y, prob_previsao)
print('roc_auc', roc_auc_score(teste_y, prob_previsao))

plt.subplots(1, figsize=(5, 5))
plt.title('Curva ROC')
plt.plot(tfp, tvp)
plt.xlabel('Especifidade')
plt.ylabel('Sensibilidade')
plt.plot([0, 1], ls="--", c='red')  # plotando linha guia pontilhada vermelha
plt.plot([0, 0], [1, 0], ls="--", c='green'), plt.plot(
    [1, 1], ls="--", c='green'
)  # plotando linha guia pontilhada verde
plt.show()


andre=27081984
base_teste = pd.read_excel('/home/andre/gdrive/DataScience/documents/desenvolvimento/fenix/base_teste/base_test.xlsx')
base_teste = pd.DataFrame(base_teste)
teste=base_teste[[ 'score','chance_de_pagamento', 
                    'risco_de_inadimplencia',
                    'capacidade_mensal_de_pagamento',
                    'comprometimento_mensal_estimado',
                    'renda_estimada'
                    ] ]  

modelo.predict(teste)

