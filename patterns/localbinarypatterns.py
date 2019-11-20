# Importa as bibliotecas necessárias.
from skimage import feature
import numpy as np

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# Guarda o número de pontos e o raio.
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# Computa a representação LBP da imagem e
		# usa a representação para gerar um histograma de padrões.
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# Normaliza o histograma.
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# Retorna o histograma de LBP.
		return hist