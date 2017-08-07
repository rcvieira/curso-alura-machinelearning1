# Eh gordinho? Tem perna curta? Late?
porco1 = [1, 1, 0]
porco2 = [1, 0, 0]
porco3 = [1, 0, 0]

cachorro4 = [1, 1, 1]
cachorro5 = [0, 1, 1]
cachorro6 = [1, 0, 1]

dados = [porco1, porco2, porco3, cachorro4, cachorro5, cachorro6]

marcacoes = [1, 1, 1, -1, -1, -1]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

misterioso1 = [1, 0, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [1, 1, 1]

teste = [misterioso1, misterioso2, misterioso3]

resultado = modelo.predict(teste)


marcacoes_teste = [-1, 1, -1]

diferencas = marcacoes_teste - resultado
acertos = [d for d in diferencas if d == 0]

total_de_acertos = len(acertos)
total_de_elementos = len(marcacoes_teste)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(resultado)
print(diferencas)
print (taxa_de_acerto)