from collections import Counter
import pandas as pd

df = pd.read_csv('dados_buscas.csv')
#df = pd.read_csv('buscas2.csv')

X_df = df[['home','busca','logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.8
porcentagem_teste = 0.1

tamanho_de_treino = int(len(Y) * porcentagem_treino)
tamanho_de_teste = int(len(Y) * porcentagem_teste)
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

fim_de_teste = tamanho_de_treino + tamanho_de_teste

teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, 
	teste_dados, teste_marcacoes):
	modelo.fit(treino_dados, treino_marcacoes)
	resultado = modelo.predict(teste_dados)

	acertos = resultado == teste_marcacoes
	total_de_acertos = sum(acertos)
	total_de_elementos = len(teste_dados)
	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
	print("Taxa de acerto do algoritmo {0}: {1}".format(nome,taxa_de_acerto))
	return taxa_de_acerto


from sklearn.naive_bayes import MultinomialNB
modeloMultinomialNB = MultinomialNB()
resultadoMultinomialNB = fit_and_predict("MultinomialNB", modeloMultinomialNB, treino_dados, treino_marcacoes, 
	teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoostClassifier = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoostClassifier, treino_dados, treino_marcacoes, 
	teste_dados, teste_marcacoes)

# Executa a validação para o vencedor
if resultadoMultinomialNB > resultadoAdaBoost:
	vencedor = modeloMultinomialNB
else:
	vencedor = modeloAdaBoostClassifier


resultado = vencedor.predict(validacao_dados)
acertos = resultado == validacao_marcacoes
total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

msg = "Taxa de acerto do vencedor no mundo real: {0}".format(taxa_de_acerto)
print(msg)


#Calculando a taxa de acerto base
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acertos_base = 100.0 * acerto_base / len(validacao_marcacoes)

print("Taxa de acerto base: %f" % taxa_de_acertos_base)
print("Total de teste: %d" % len(validacao_dados))