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
# On affiche ensuite le nuage des individus projeté sur les deux premiers axes factoriels, en colorant chaque cellule selon sa 
# classe (rouge pour maligne, bleu pour bénigne). Les deux classes apparaissent-elles séparées ?
titre = "2 Le nuage des individus projeté sur les deux premiers axes"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Cellules : " + titre)
for i in range(len(y)):
    xx, yy = cellulesDansNouveauRepere[i,0], cellulesDansNouveauRepere[i,1]
    ax.scatter(xx, yy, c=coul[i])
ax.set_xlabel(legende_axes[0])
ax.set_ylabel(legende_axes[1])
fig.savefig(titre, dpi=150, bbox_inches="tight")


titre = "3 Le nuage des individus projeté sur les deux axes suivants"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Cellules : " + titre)
for i in range(len(y)):
    xx, yy = cellulesDansNouveauRepere[i,2], cellulesDansNouveauRepere[i,3]
    ax.scatter(xx, yy, c=coul[i])
ax.set_xlabel(legende_axes[2])
ax.set_ylabel(legende_axes[3])
fig.savefig(titre, dpi=150, bbox_inches="tight")


titre = "4 Le nuage des individus projeté sur les trois premiers axes"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Cellules : " + titre)
for i in range(len(y)): xx, yy, zz = cellulesDansNouveauRepere[i,0], cellulesDansNouveauRepere[i,1],cellulesDansNouveauRepere[i,2]
ax.scatter(xx, yy, zz, c=coul[i])
ax.set_xlabel(legende_axes[0])
ax.set_ylabel(legende_axes[1])
ax.set_zlabel(legende_axes[2])
fig.savefig(titre, dpi=150, bbox_inches="tight")