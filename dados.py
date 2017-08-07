import csv

def carregar_acessos():

	X = []
	Y = []

	arquivo = open('dados_acesso.csv', 'rt', encoding='UTF8')
	leitor = csv.reader(arquivo)

	next(leitor)

	for acessou_home,acessou_como_funciona,acessou_contato,comprou in leitor:
		X.append([int(acessou_home),int(acessou_como_funciona),int(acessou_contato)])
		Y.append(int(comprou))

	return X, Y