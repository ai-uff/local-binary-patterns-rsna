# USO: Passar como argumento o diretório de imagens para treinamento e teste.
# python predict.py --training images/training --testing images/testing

# Importa as bibliotecas necessárias
from patterns.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os

# Parseia os argumentos da linha de comando.
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=True,
	help="path to the tesitng images")
args = vars(ap.parse_args())

# Inicializa o descritor LBP (Local Binary Patterns)
# com os dados e a lista de labels
desc = LocalBinaryPatterns(24, 8)
data = [] # será armazedado para cada imagem sua descrição LBP.
labels = [] # será armazedado os rótulos para classficação.

# Itera sobre o conjunto de treinamento de imagens (training images)
for imagePath in paths.list_images(args["training"]):
	# Carrega a imagem, converte para escala cinza e descreve isso com LBP.
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Gera as imagens LBP e Histograma
	hist = desc.describe(gray) # Retorna o resultado do descritor LBP.

	# Extrai os labels dos paths da imagem, depois atualiza a
	# lista de dados e labels inicializada anteriormente.
	labels.append(imagePath.split(os.path.sep)[-2])
	data.append(hist)

# Resultado
# data = [hist_1, ... , ..., hist_8]
# labels = ['epidural', ..., 'intraparenchymal']

# Treina um classificador linear SVM nos dados
# Para maiores informações: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
model = LinearSVC(C=50.0, random_state=42, max_iter=1500) # Cria modelo com os parâmetros citados.
model.fit(data, labels) # Treina o modelo, como parâmetro recebe os dados e os labels (rótulos das classes)

# Resultado
# Modelo está treinado

# Para cada descriotor previamente extraido dos dados deve existir um label que os identifique.
# Descritor da imagem 1 será classificado como rótulo e assim por diante.

# Itera sobre as imagens de teste
for imagePath in paths.list_images(args["testing"]):
	# Carrega imagem, converte isso para escala cinza e descreve isso com LBP.
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# Classifica a imagem entre os labels existentes.
	prediction = model.predict(hist.reshape(1, -1))

	# Mostra a imagem e a previsão
	cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		0.4, (0, 0, 255), 1)
	cv2.imshow("Prediction", image)
	cv2.waitKey(0)
