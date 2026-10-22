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

titre = "4 Projection des variables sur les deux premiers axes factoriels"
projectionsVars = monACP.components_.T * np.sqrt(monACP.explained_variance_)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Vins : " + titre)
ax.set_xlim([-1, 1]) ; ax.set_ylim([-1, 1])
ax.axhline(0, color='lightgray', lw=0.5) ;ax.axvline(0, color='lightgray', lw=0.5)
ax.add_artist(plt.Circle((0, 0), 1, color='b', fill=False))
# projectionsVars : coordonnées de chacune des variables dans le nouveau repère
for i, nom in enumerate(nomsVars):
    x, y = projectionsVars[i, 0], projectionsVars[i, 1]
    ax.arrow(0, 0, x, y, shape='full', lw=1, length_includes_head=True,head_width=0.03, head_length=0.05)
    ax.text(x, y, nom, fontsize=7)
ax.set_xlabel(legende_axes[0])
ax.set_ylabel(legende_axes[1])
#afficher la figure 3D et l'enregistrer
plt.show()
fig.savefig(titre, dpi=150, bbox_inches="tight")



titre = "5 Projection des variables sur les deux premiers axes factoriels"
projectionsVars = monACP.components_.T * np.sqrt(monACP.explained_variance_)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Vins : " + titre)
ax.set_xlim([-1, 1]) ; ax.set_ylim([-1, 1])
ax.axhline(0, color='lightgray', lw=0.5) ;ax.axvline(0, color='lightgray', lw=0.5)
ax.add_artist(plt.Circle((0, 0), 1, color='b', fill=False))
# projectionsVars : coordonnées de chacune des variables dans le nouveau repère
for i, nom in enumerate(nomsVars):
    x, y = projectionsVars[i, 2], projectionsVars[i, 3]
    ax.arrow(0, 0, x, y, shape='full', lw=1, length_includes_head=True,head_width=0.03, head_length=0.05)
    ax.text(x, y, nom, fontsize=7)
ax.set_xlabel(legende_axes[2])
ax.set_ylabel(legende_axes[3])
#afficher la figure 3D et l'enregistrer
plt.show()
fig.savefig(titre, dpi=150, bbox_inches="tight")



