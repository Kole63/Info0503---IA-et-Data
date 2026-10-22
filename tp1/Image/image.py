import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_olivetti_faces
visages = fetch_olivetti_faces()
X = visages.data            # (400, 4096)
personnes = visages.target  # (400,) : 40 personnes, 10 visages chacune
print("Dimensions du tableau :", X.shape)
print("Effectifs par personne :", np.bincount(personnes))
titre = "0 Les dix premiers visages"
fig, axes3 = plt.subplots(2, 5, figsize=(12, 5))
for k, ax3 in enumerate(axes3.ravel()):
    vImage = X[k].reshape(64, 64)
    ax3.imshow(vImage, cmap="gray")
    ax3.set_title(f"Visage {k+1})")
    ax3.axis("off")
fig.suptitle(titre)