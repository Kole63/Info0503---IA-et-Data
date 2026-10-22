
#http://cedric.cnam.fr/vertigo/Cours/ml/tpAcp.html
#https://scikit-learn.org/stable/index.html
#https://moonbooks.org/Articles/Analyse-en-composantes-principales-avec-python/
from sklearn.decomposition import PCA  #pip scikit-learn
from sklearn import preprocessing
from math import *
from random import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('VecteursMaths.csv', sep=';', decimal=',', index_col=0)
df = df[['X','Y','Z','T']]


nbIndividus = len(df)

nomsPoints = df.index
nomsVecteurs = df.columns
print(f"{nomsVecteurs=}")

titre="1 Le nuage des points sur les 2 premiers axes"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("VecteurMaths : "+titre)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
ax.plot(df['X'],df['Y'],".b")
ax.set_xlabel("Axe X")
ax.set_ylabel("Axe Y")
fig.savefig(titre, dpi=150, bbox_inches="tight")

titre="2 Le nuage des points sur les 2 derniers axes"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("VecteurMaths : "+titre)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
ax.plot(df['Z'],df['T'],".g")
ax.set_xlabel("Axe Z")
ax.set_ylabel("Axe T")
fig.savefig(titre, dpi=150, bbox_inches="tight")

titre="3 Le nuage des points sur les 3 premiers axes"
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("VecteurMaths : "+titre)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)
ax.set_box_aspect([1, 1, 1])
ax.plot(df['X'], df['Y'], df['Z'], ".b")
ax.set_xlabel("Axe X")
ax.set_ylabel("Axe Y")
ax.set_zlabel("Axe Z")
fig.savefig(titre, dpi=150, bbox_inches="tight")

vecteursCR = preprocessing.scale(df.to_numpy()) #Variables centrées réduites

monACP = PCA() #PCA(n_components=2) ou PCA(n_components=0.95) pour limiter le nb de composantes
monACP.fit(vecteursCR) # calculs

variance_ratio_ = monACP.explained_variance_ratio_
print(f"{variance_ratio_=}")

titre='4 Pourcentage d’inertie associé à chaque axe factoriel'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("VecteurMaths : "+titre)
lk=range(len(variance_ratio_))
ax.bar(lk, variance_ratio_)
ax.set_xticks(lk)
ax.set_xticklabels([f"PC{i+1}" for i in lk])
ax.set_xlabel("Composante principale")
ax.set_ylabel("% de variance expliquée")
fig.savefig(titre, dpi=150, bbox_inches="tight")

vecteursDansNouveauRepere = monACP.transform(vecteursCR) #
print(f"{vecteursDansNouveauRepere[:2,:]=}")

vecteurs = pd.DataFrame(
    vecteursDansNouveauRepere[:2,:],
    index=["PC1","PC2"],
    columns=nomsVecteurs
).T  # transposé : lignes = variables, colonnes = composantes
print("=== vecteursDansNouveauRepere ===")
print(vecteurs.to_string())


# projectionsVars est composée des coordonnées de chacune des variables dans le nouveau repère
projectionsVars = monACP.components_.T * np.sqrt(monACP.explained_variance_)


# Tracé du cercle des corrélations sur les axes 1 et 2
titre="5 Projection des variables sur les deux premiers axes factoriels"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.axhline(0, color='lightgray', lw=0.5)
ax.axvline(0, color='lightgray', lw=0.5)
cercle = plt.Circle((0, 0), 1, color='b', fill=False)
ax.add_artist(cercle)
for i, nom in enumerate(nomsVecteurs):
    x, y = projectionsVars[i, 0], projectionsVars[i, 1]
    ax.arrow(0, 0, x, y, shape='full', lw=1, length_includes_head=True, head_width=0.03, head_length=0.05)
    ax.text(x, y, nom,color='red'  if nom in ['poids','taille'] else 'blue')
ax.set_xlabel("Axe 1")
ax.set_ylabel("Axe 2")
ax.set_title("VecteurMaths : "+titre)
fig.savefig(titre, dpi=150, bbox_inches="tight")

titre="6 Projection des variables sur les deux derniers axes factoriels"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.axhline(0, color='lightgray', lw=0.5)
ax.axvline(0, color='lightgray', lw=0.5)
cercle = plt.Circle((0, 0), 1, color='b', fill=False)
ax.add_artist(cercle)
for i, nom in enumerate(nomsVecteurs):
    x, y = projectionsVars[i,2], projectionsVars[i,3]
    ax.arrow(0, 0, x, y, shape='full', lw=1, length_includes_head=True, head_width=0.03, head_length=0.05)
    ax.text(x, y, nom,color='red'  if nom in ['poids','taille'] else 'blue')
ax.set_xlabel("Axe 3")
ax.set_ylabel("Axe 4")
ax.set_title("VecteurMaths : "+titre)
fig.savefig(titre, dpi=150, bbox_inches="tight")





titre="7 projection des individus sur les deux premiers axes factoriels"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("VecteurMaths : "+titre)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.plot(vecteursDansNouveauRepere[:,0],vecteursDansNouveauRepere[:,1],'b+')
ax.set_xlabel("Axe 1")
ax.set_ylabel("Axe 2")
fig.savefig(titre, dpi=150, bbox_inches="tight")

titre="8 projection des individus sur les deux derniers axes factoriels"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("VecteurMaths : "+titre)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.plot(vecteursDansNouveauRepere[:,2],vecteursDansNouveauRepere[:,3],'g+')
ax.set_xlabel("Axe 3")
ax.set_ylabel("Axe 4")
fig.savefig(titre, dpi=150, bbox_inches="tight")

titre="9 projection des individus sur les trois premiers axes factoriels"
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("VecteurMaths : "+titre)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_zlim(-5,5)
ax.set_box_aspect([1, 1, 1])
ax.plot(vecteursDansNouveauRepere[:,0],vecteursDansNouveauRepere[:,1],vecteursDansNouveauRepere[:,2], ".b")
ax.set_xlabel("Axe 1")
ax.set_ylabel("Axe 2")
ax.set_zlabel("Axe 3")
fig.savefig(titre, dpi=150, bbox_inches="tight")


ratios = monACP.explained_variance_ratio_
composantes = [f"PC{i+1}" for i in range(monACP.n_components_)]

# 2. Tableau des vecteurs propres
#    lignes = composantes, colonnes = variables initiales
vecteurs = pd.DataFrame(
    monACP.components_.round(3),
    index=composantes,
    columns=nomsVecteurs
).T  # transposé : lignes = variables, colonnes = composantes
print("=== Vecteurs propres (coordonnées par variable) ===")
print(vecteurs.to_string())


# ------------------------------------------------------------------
# Approximation des données à partir de seulement PC1 et PC2
#    On calcule leurs coordonnées et on décentre (puisque l'on avait centré réduit)
# ------------------------------------------------------------------
ecarts_types = vecteursCR.std(axis=0)      #Tableau de moy et ecart types
moyennes = vecteursCR.mean(axis=0)

XYZTApprox =( vecteursDansNouveauRepere[:, :2]     @  monACP.components_[:2, :]) * ecarts_types+ moyennes
T_estime = XYZTApprox[:, 3]

T_reel = vecteursCR[:, 3] * ecarts_types[3] + moyennes[3]

titre= "10 Qualité de l'estimation de T par ACP (2 composantes)"
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(T_reel, T_estime, c="steelblue", s=40, alpha=0.8)
borne = max(abs(T_reel).max(), abs(T_estime).max()) * 1.1
ax.plot([-borne, borne], [-borne, borne], "r--", label="y = x")
ax.set_xlabel("T réel")
ax.set_ylabel("T estimé à partir de PC1 et PC2")
ax.set_title("VecteurMaths : "+titre)
ax.legend()
ax.grid(True, linestyle=":", alpha=0.5)
fig.savefig(titre, dpi=150, bbox_inches="tight")
plt.tight_layout()
plt.show()