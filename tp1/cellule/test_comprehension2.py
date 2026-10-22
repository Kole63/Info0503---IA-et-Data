import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
# Génération de données selon une loi normale tridimensionnelle
mesVectAlea = np.random.randn(500, 3)
otherVectAlea = np.random.randn(500, 3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mesVectAlea[:,0], mesVectAlea[:,1], mesVectAlea[:,2])
plt.title("Données initiales")
plt.show()
# Appliquez ensuite l'ACP à ces données :
monACP = PCA(n_components=3)
monACP.fit(mesVectAlea)
print("Pourcentage de variance expliquée : ",monACP.explained_variance_ratio_)
print("Composantes principales : ")
print(monACP.components_)


s1 = np.array([[3,0,0], [0,1,0], [0,0,0.2]])  # matrice de déformation
r1 = np.array([[0.36,0.48,-0.8], [-0.8,0.6,0], [0.48,0.64,0.6]])  # matrice de rotation
mesVectDeformes = mesVectAlea.dot(s1).dot(r1)

monACP2 = PCA(n_components=3)
monACP2.fit(mesVectDeformes)
print("Pourcentage de variance expliquée : ",monACP2.explained_variance_ratio_)
print("Composantes principales : ")
print(monACP2.components_)

