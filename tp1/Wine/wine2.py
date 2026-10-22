from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
vins = load_wine()
X = vins.data     # (569, 30)
y = vins.target   # (569,) : 0 = maligne, 1 = bénigne
coul = ["red" if yy == 0 else "blue" for yy in y]
nomsVars = vins.feature_names
print("Dimensions du tableau :", X.shape)
print("Effectifs par valeur de y :", [y.tolist().count(k) for k in range(2)])
print(f"{len(nomsVars)} variables sur {len(y)} vins.")

pcaBrut = PCA().fit(X)
print(pcaBrut.explained_variance_ratio_[:5])
wineCR = preprocessing.scale(X)
monACP = PCA().fit(wineCR)
monACPNonCR = PCA().fit(X)

print(monACPNonCR.explained_variance_ratio_[:4]) #le premier axe capture 99,8 % de l'inertie
monACPNonCR.components_[0] # La dernière composante de cet axe a un coefficient presque à 1
print(X)  

wineCR = preprocessing.scale(X)
monACP = PCA().fit(wineCR)

vinsDansNouveauRepere = monACP.transform(wineCR)
legende_axes = [f"Axe {k+1} ({monACP.explained_variance_ratio_[k]:.1%})" for k in 
range(4)]

titre = "2 Projection des vins dans le nouveau repère nuage de points"

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111) 
ax.set_title("Vins : " + titre)
for i in range(len(y)):
    xx, yy = vinsDansNouveauRepere[i,0], vinsDansNouveauRepere[i,1]
    ax.scatter(xx, yy, c=coul[i])
ax.set_xlabel(legende_axes[0])
ax.set_ylabel(legende_axes[1])
fig.savefig(titre, dpi=150, bbox_inches="tight")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111) 
ax.set_title("Vins 3: " + titre)
for i in range(len(y)):
    xx, yy = vinsDansNouveauRepere[i,2], vinsDansNouveauRepere[i,3]
    ax.scatter(xx, yy, c=coul[i])
ax.set_xlabel(legende_axes[2])
ax.set_ylabel(legende_axes[3])
fig.savefig("Vins nuage 2", dpi=150, bbox_inches="tight")

