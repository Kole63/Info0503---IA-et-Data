#################################
#
#Le code source de l'exemple du
#cours : mammiferes.py
#
#################################
# Données : Allison T., D. Cicchetti. Sleep in mammals: ecological and constitutional correlates. Science, 194: 732–734 (1976).
#http://cedric.cnam.fr/vertigo/Cours/ml/tpAcp.html
#https://scikit-learn.org/stable/index.html
#https://moonbooks.org/Articles/Analyse-en-composantes-principales-avec-python/
from sklearn.decomposition import PCA  #pip scikit-learn
from sklearn import preprocessing
from math import * ; import matplotlib.pyplot as plt
import numpy as np  ;import pandas as pd

#Lecture du fichier csv dans un dataframe panda pour faire éventuellement du traitement de données
df = pd.read_csv('mammiferes.csv', sep=';', decimal=',', index_col=0)
#df = df.query("poids >= 90 & genre==0 & hauteur_cervicale>0")
#df = df.drop(columns=['age','genre','tranche_d_age'])

nomsIndividus = df.index
nomsVars = df.columns
print(f"{len(nomsVars)} variables sur {len(nomsIndividus)} espèces de mammifères.")
# mammiferesCR est le Tableau NumPy des mesures numériques centrées réduites
mammiferesCR = preprocessing.scale(df.to_numpy())

# "l'Entrainement", le calcuļ eņ fait, de l'ACP par scikitlearn se fait classiquement
monACP = PCA().fit(mammiferesCR) #PCA(n_components=2) ou PCA(n_components=0.95) pour limiter le nb de composantes

print("\nVariance expliquée par axe :")
for i, (vp, part) in enumerate(zip(monACP.explained_variance_, monACP.explained_variance_ratio_)):
    print(f"  Axe {i+1} : valeur propre = {vp:6.3f} | part = {part:6.2%} "
          f"| cumul = {monACP.explained_variance_ratio_[:i].sum():6.2%}")

#Légende des axes factoriels pour les graphiques
legende_axes=[f"Axe {k+1} ({monACP.explained_variance_ratio_[k]:.1%}" for k in range(4)]

titre='1 Pourcentage d’inertie associé à chaque axe factoriel'
seuil=0.95
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
indices = range(len(monACP.explained_variance_ratio_)) #liste des indices des axes
ax.bar(indices, monACP.explained_variance_ratio_, color="steelblue", label="Variance expliquée")
ax.plot(indices, monACP.explained_variance_ratio_.cumsum(), "o-", color="crimson", label="Variance cumulée")
ax.axhline(seuil, ls="--", color="grey", label=f"Seuil {int(seuil*100)} %")
ax.set_xticks(indices)
ax.set_xticklabels([f"PC{i+1}" for i in indices])
ax.set_xlabel("Composante principale")
ax.set_ylabel("% de variance expliquée")
ax.set_title(titre)
ax.legend()
fig.savefig(titre, dpi=150, bbox_inches="tight")

#Les coordonnées des individus dans le repère formé par les axes factoriels
mammiferesDansNouveauRepere = monACP.transform(mammiferesCR) #

print(f"Coordonnées des mammifères dans le nouveau repère :")
print("                         ",end="")
for j in range(len(nomsVars)):
        print(f" Axe {j+1} ",end=" ")
print()
for label, ligne in zip(nomsIndividus, mammiferesDansNouveauRepere):
    print(f"  {label[:20]:<20} :",end=" ")
    for valeur in ligne:
        print(f"{valeur:+.4f}",end=" ")
    print()

titre="2 Le nuage des individus projeté sur les deux premiers axes"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Mammifères : "+titre)
for i,nom in enumerate(nomsIndividus):
        x, y = mammiferesDansNouveauRepere[i,0], mammiferesDansNouveauRepere[i,1]
        ax.scatter(x, y)
        ax.text(x, y, nom, ha='center')
ax.set_xlabel(legende_axes[0])
ax.set_ylabel(legende_axes[1])
fig.savefig(titre, dpi=150, bbox_inches="tight")

titre="3 Le nuage des individus projeté sur les deux axes suivants"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Mammifères : "+titre)
for i,nom in enumerate(nomsIndividus):
        x, y = mammiferesDansNouveauRepere[i,0], mammiferesDansNouveauRepere[i,1]
        ax.scatter(x, y)
        ax.text(x, y, nom, ha='center')
ax.set_xlabel(legende_axes[2])
ax.set_ylabel(legende_axes[3])
fig.savefig(titre, dpi=150, bbox_inches="tight")


titre="4 Le nuage des individus projeté sur les trois premiers axes"
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Mammifères : "+titre)
for i,nom in enumerate(nomsIndividus):
        x, y, z = mammiferesDansNouveauRepere[i,0], mammiferesDansNouveauRepere[i,1], mammiferesDansNouveauRepere[i,2]
        ax.scatter(x, y, z)
        ax.text(x, y, z, nom, ha='center')
ax.set_xlabel(legende_axes[0])
ax.set_ylabel(legende_axes[1])
ax.set_zlabel(legende_axes[2])
fig.savefig(titre, dpi=150, bbox_inches="tight")



# projectionsVars est composée des coordonnées de chacune des variables dans le nouveau repère
projectionsVars = monACP.components_.T * np.sqrt(monACP.explained_variance_)

titre="5 Projection des variables sur les deux premiers axes factoriels"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Mammifères : "+titre)
ax.set_xlim([-1, 1]) ; ax.set_ylim([-1, 1])
ax.axhline(0, color='lightgray', lw=0.5) ; ax.axvline(0, color='lightgray', lw=0.5)
ax.add_artist(plt.Circle((0, 0), 1, color='b', fill=False))

for i, nom in enumerate(nomsVars):
    x, y = projectionsVars[i, 0], projectionsVars[i, 1]
    ax.arrow(0, 0, x, y, shape='full', lw=1, length_includes_head=True, head_width=0.03, head_length=0.05)
    ax.text(x, y, nom, color='red' if nom in ['poids','taille'] else 'blue')
ax.set_xlabel(legende_axes[0])
ax.set_ylabel(legende_axes[1])
fig.savefig(titre, dpi=150, bbox_inches="tight")

titre="6 Projection des variables sur les deux derniers axes factoriels"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Mammifères : "+titre)
ax.set_xlim([-1, 1]) ; ax.set_ylim([-1, 1])
ax.axhline(0, color='lightgray', lw=0.5) ; ax.axvline(0, color='lightgray', lw=0.5)
ax.add_artist(plt.Circle((0, 0), 1, color='b', fill=False))

for i, nom in enumerate(nomsVars):
    x, y = projectionsVars[i, 2], projectionsVars[i, 3]
    ax.arrow(0, 0, x, y, shape='full', lw=1, length_includes_head=True, head_width=0.03, head_length=0.05)
    ax.text(x, y, nom, color='red' if nom in ['poids','taille'] else 'blue')
ax.set_xlabel(legende_axes[2])
ax.set_ylabel(legende_axes[3])
fig.savefig(titre, dpi=150, bbox_inches="tight")

composantes = [f"PC{i+1}" for i in range(monACP.n_components_)]

# Tableau des vecteurs propres
#    lignes = composantes, colonnes = variables initiales
vecteurs = pd.DataFrame(
    monACP.components_.round(3),
    index=composantes,
    columns=nomsVars
).T  # transposé : lignes = variables, colonnes = composantes
print("=== Vecteurs propres (coordonnées par variable) ===")
print(vecteurs.to_string())

plt.show()