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
# ou PCA(n_components=2) pour limiter le nombre de composantes,
# ou PCA(n_components=0.95) pour conserver 95 % de la variance
# On peut ensuite afficher les premiers résultats du calcul selon la manière habituelle :
print("\nVariance expliquée par axe :")
for i, (vp, part) in enumerate(zip(monACP.explained_variance_, monACP.explained_variance_ratio_)):
    print(f"  Axe {i+1} : valeur propre = {vp:6.3f} | part = {part:6.2%} "f"| cumul = {monACP.explained_variance_ratio_[:i].sum():6.2%}")
titre = '1 Pourcentage d’inertie associé à chaque axe factoriel'
seuil = 0.95
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
indices = range(len(monACP.explained_variance_ratio_))
ax.bar(indices, monACP.explained_variance_ratio_, color="steelblue", label="Variance expliquée")
ax.plot(indices, monACP.explained_variance_ratio_.cumsum(), "o-", color="crimson", 
label="Variance cumulée")
ax.axhline(seuil, ls="--", color="grey", label=f"Seuil {int(seuil*100)} %")
ax.set_xticks(indices)
ax.set_xticklabels([f"PC{i+1}" for i in indices])
ax.set_xlabel("Composante principale")
ax.set_ylabel("% de variance expliquée")
ax.set_title(titre)
ax.legend()
fig.savefig(titre, dpi=150, bbox_inches="tight")