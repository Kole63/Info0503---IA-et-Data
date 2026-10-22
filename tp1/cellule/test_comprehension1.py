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

monACP2 = PCA(n_components=3)
monACP2.fit(otherVectAlea)
print("Pourcentage de variance expliquée : ",monACP2.explained_variance_ratio_)
print("Composantes principales : ")
print(monACP2.components_)
