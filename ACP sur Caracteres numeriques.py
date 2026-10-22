# -*- coding: utf-8 -*-
"""
Analyse en Composantes Principales sur le jeu load_digits() de scikit-learn
(jeu complet : 1797 images de chiffres manuscrits, 64 pixels par image).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# 1. Chargement des données
# ----------------------------------------------------------------------
chiffres = load_digits()  #Exemple proposé par sklearn
X_complet, y_complet = chiffres.data, chiffres.target
masque = np.isin(y_complet, [0,1,2,3,4,5,6,7,8,9])
X = X_complet[masque]
y = y_complet[masque]

print("Dimensions du tableau :", X.shape)
print("Effectifs par chiffre :", [y.tolist().count(k) for k in range(10)])
print(X)

#monScaler=StandardScaler()  #On crée une instance pour conserver les paramètres et pouvoir "revenir en arrière"
#CXR = monScaler.fit_transform(X) #Note StandartScaler gère les pixels "constants"

CXR=X  #On ne centre réduit pas mais en faite l'ACP centre automatiquement'

monACP = PCA(n_components=64).fit(CXR)
coordonneesDansNouveauRepere = monACP.transform(CXR)

variance_expliquee = monACP.explained_variance_ratio_
valeurs_propres = monACP.explained_variance_

print("\nVariance expliquée par axe :")
for i, (vp, part) in enumerate(zip(monACP.explained_variance_, monACP.explained_variance_ratio_)):
    print(f"  Axe {i+1} : valeur propre = {vp:6.3f} | part = {part:6.2%} "
          f"| cumul = {monACP.explained_variance_ratio_[:i].sum():6.2%}")

titre='1 Pourcentage d’inertie associé à chaque axe factoriel'
indices = range(len(monACP.explained_variance_ratio_))
fig, ax = plt.subplots(figsize=(8, 5))
x_ax = np.arange(1, len(valeurs_propres) + 1)
ax.bar(x_ax, variance_expliquee, color="steelblue", label="Variance expliquée")
ax.plot(x_ax, variance_expliquee.cumsum(), "o-", color="crimson",
        label="Variance cumulée", markersize=3)
ax.axhline(0.80, ls="--", color="grey", label="Seuil 80 %")
ax.set_xticks(indices)
ax.set_xticklabels([f"{i+1}" for i in indices])
ax.set_xlabel("Composante principale")
ax.set_ylabel("% de variance expliquée")
ax.set_title(titre)
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(titre, dpi=150, bbox_inches="tight")

titre="2. Quelques caractères de notre jeu"
fig, axes = plt.subplots(5, 20, figsize=(12, 3))
for k, ax in enumerate(axes.ravel()):
    ax.imshow(X[k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
    ax.set_title(f"Im{k}:{y[k]}", fontsize=9)
    ax.axis("off")
fig.suptitle(titre)
fig.savefig(titre+".png", dpi=150, bbox_inches="tight")
plt.tight_layout()
plt.show()

titre="3. Projection des 1797 images"
fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
axes = axes.ravel() # liste à une dimension des axes
for i, ax in enumerate(axes):
    scatter = ax.scatter(coordonneesDansNouveauRepere[:, 2*i], coordonneesDansNouveauRepere[:, 2*i+1],
                         c=y, cmap="tab10", s=12, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"Axe {2*i + 1} ({variance_expliquee[2*i]:.1%})")
    ax.set_ylabel(f"Axe {2*i + 2} ({variance_expliquee[2*i+1]:.1%})")
    ax.set_title(f"Projections sur les axes {2*i+1} et {2*i+2}")
fig.colorbar(scatter, ax=axes, label="Chiffre", shrink=0.85)
fig.suptitle(titre, fontsize=13)
plt.show()
fig.savefig(titre+".png", dpi=150, bbox_inches="tight")


n_axes = 20
part_variance=variance_expliquee[:n_axes].sum()
print(f"\nReconstruction avec {n_axes} axes ({part_variance:.1%} de la variance expliquée) :")

# Reconstruction dans l'espace centré-réduit :
#X_std_reconstruit = (coordonneesDansNouveauRepere[:, :n_axes] @ monACP.components_[:n_axes]                     + monACP.mean_)
# X_reconstruit = monScaler.inverse_transform(X_std_reconstruit)
X_reconstruit = (coordonneesDansNouveauRepere[:, :n_axes] @ monACP.components_[:n_axes]+ monACP.mean_)

# Les axes factoriels à 64 dimensions affichés comme des caractères --------------
# # Si c'était Centré réduit, il faudrait remultiplier par l'échelle''
#poids_pixels = monACP.components_[:n_axes] * monScaler.scale_
poids_pixels = monACP.components_[:n_axes]

titre="3 Les premiers axes factoriels vus comme des caractères"
vmax_axes = np.max(np.abs(poids_pixels))
fig, axes_fig = plt.subplots(2, 5, figsize=(12, 5.5))
for k, ax in enumerate(axes_fig.ravel()):
    ax.imshow(poids_pixels[k].reshape(8, 8), cmap="RdBu_r",
              vmin=-vmax_axes, vmax=vmax_axes)
    ax.set_title(f"Axe {k+1} ({variance_expliquee[k]:.1%})", fontsize=10)
    ax.axis("off")
fig.suptitle(titre+"(rouge = poids positif, bleu = poids négatif)")
plt.tight_layout()
plt.show()
fig.savefig(titre+".png", dpi=150, bbox_inches="tight")

titre="4 Caractères reconstruits avec leur combinaison linéaire"
# Formule utilisée (dans l'espace centré-réduit, puis remise à l'échelle) :
#   reconstruit_i = moyenne_pixels
#                   + somme_{k=1..10} coordonneesDansNouveauRepere[i, k-1] * axe_k * sigma
n_exemples = 6
print(f"\nCombinaisons linéaires utilisées ({n_exemples} premiers exemples) :")
for i in range(n_exemples):
    termes = " ".join(
        f"{coordonneesDansNouveauRepere[i, j]:+.2f}*A{j+1}" for j in range(n_axes)
    )
    print(f"  image {i} (chiffre {y[i]}) : moyenne + {termes}")

fig, axes = plt.subplots(2, n_exemples, figsize=(2.8 * n_exemples, 6))
for k in range(n_exemples):
    axes[0, k].imshow(X[k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
    axes[0, k].set_title(f"image {k} - chiffre {y[k]}", fontsize=9)
    axes[0, k].axis("off")
    axes[1, k].imshow(X_reconstruit[k].reshape(8, 8), cmap="gray_r",
                      vmin=0, vmax=16)
    coeff = ", ".join(f"{coordonneesDansNouveauRepere[k, j]:.1f}" for j in range(n_axes))
    axes[1, k].set_title(f"comb. lin. : ({coeff})",
                         fontsize=7, pad=2)
    axes[1, k].axis("off")
fig.suptitle(titre+f"""({part_variance:.1%} de la variance)
             titre du bas : coefficients sur les axes 1 à 10""")
plt.tight_layout()
plt.show()
fig.savefig(titre+".png", dpi=150, bbox_inches="tight")
