# Primeira abordagem foi 90% para treino e 10% para teste
# resultado 88%
from dados import carregar_acessos

dados,marcacoes = carregar_acessos()

treino_dados = dados[:90]
treino_marcacoes = marcacoes[:90]

teste_dados = dados[-9:]
teste_marcacoes = marcacoes[-9:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

diferencas = resultado - teste_marcacoes
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(taxa_de_acerto)
print(total_de_elementos)

