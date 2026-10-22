from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X = cancer.data     # (569, 30)
y = cancer.target   # (569,) : 0 = maligne, 1 = bénigne
coul = ["red" if yy == 0 else "blue" for yy in y]
nomsVars = cancer.feature_names
print("Dimensions du tableau :", X.shape)
print("Effectifs par valeur de y :", [y.tolist().count(k) for k in range(2)])
print(f"{len(nomsVars)} variables sur {len(y)} cellules.")

# pcaBrut = PCA().fit(X)
# print(pcaBrut.explained_variance_ratio_[:5])
cellulesCR = preprocessing.scale(X)
monACP = PCA().fit(cellulesCR)


cellulesDansNouveauRepere = monACP.transform(cellulesCR)
# On peut afficher les coordonnées ligne par ligne (une ligne par cellule, une colonne par axe factoriel) :
print("Coordonnées des cellules dans le nouveau repère :")
print("                         ", end="")
for j in range(4):
    print(f" Axe {j+1} ", end=" ")
print()
for k, ligne in enumerate(cellulesDansNouveauRepere):
    print(f"  {k:4} :", end=" ")
    for valeur in ligne:
        print(f"{valeur:+.4f}", end=" ")
    print()
# La légende des axes factoriels pour les graphiques intègre la part de variance de chaque axe :
legende_axes = [f"Axe {k+1} ({monACP.explained_variance_ratio_[k]:.1%})" for k in 
range(4)]



# projectionsVars : coordonnées de chacune des variables dans le nouveau repère
projectionsVars = monACP.components_.T * np.sqrt(monACP.explained_variance_)
titre = "5 Projection des variables sur les deux premiers axes factoriels"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Cellules : " + titre)
ax.set_xlim([-1, 1]) ; ax.set_ylim([-1, 1])
ax.axhline(0, color='lightgray', lw=0.5) ; ax.axvline(0, color='lightgray', lw=0.5)
ax.add_artist(plt.Circle((0, 0), 1, color='b', fill=False))
for i, nom in enumerate(nomsVars):
    x, y = projectionsVars[i, 0], projectionsVars[i, 1]
    ax.arrow(0, 0, x, y, shape='full', lw=1, length_includes_head=True,head_width=0.03, head_length=0.05)
    ax.text(x, y, nom, fontsize=7)
ax.set_xlabel(legende_axes[0])
ax.set_ylabel(legende_axes[1])
fig.savefig(titre, dpi=150, bbox_inches="tight")