# Classification d’images CIFAR-10 avec PyTorch

Ce projet entraîne plusieurs réseaux de neurones convolutionnels (CNN) pour classifier des images du dataset CIFAR-10.

L’objectif du projet est de comprendre :

- l’entraînement d’un réseau convolutionnel
- l’impact de différentes architectures CNN
- l’évaluation des performances d’un modèle de classification
- la visualisation des représentations apprises par le réseau
- l’utilisation de techniques avancées d’entraînement avec PyTorch

Le projet est implémenté avec la bibliothèque **PyTorch**.

Au-delà d’un entraînement classique, ce projet explore également plusieurs techniques utilisées dans les pipelines modernes d’apprentissage profond :

- **data augmentation** pour améliorer la généralisation
- utilisation de l’optimiseur **AdamW**
- **label smoothing** dans la fonction de perte
- **learning rate warmup**
- **Cosine Annealing learning rate scheduler**
- **gradient clipping** pour stabiliser l’apprentissage

L’objectif est d’aller plus loin qu’un simple exemple de classification et d’expérimenter différentes stratégies d’entraînement afin d’observer leur impact sur les performances du modèle.

# Dataset

Le dataset utilisé est **CIFAR-10**, un dataset très utilisé pour les tâches de classification d’images.

Caractéristiques du dataset :

- 50 000 images d’entraînement  
- 10 000 images de test  
- images en couleur (RGB)  
- taille : **32 × 32 pixels**
- **10 classes d’objets**

Les classes sont :

[ airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck ]

Chaque image représente un objet appartenant à l’une de ces catégories.


# Prétraitement des données

Pour améliorer la généralisation du modèle, plusieurs techniques de **data augmentation** sont utilisées :

- **Random Horizontal Flip** : retourne aléatoirement les images horizontalement pour augmenter la diversité des données.
- **Random Crop avec padding** : recadre aléatoirement une partie de l’image après avoir ajouté un léger padding, ce qui rend le modèle plus robuste aux variations de position.
- **Random Rotation** : applique une petite rotation aléatoire aux images afin d’améliorer la capacité du modèle à reconnaître les objets sous différents angles.
- **Color Jitter** : modifie légèrement la luminosité et le contraste pour rendre le modèle plus robuste aux variations d’éclairage.

---

# Architecture du modèle

Le modèle utilisé est un **réseau convolutionnel modulaire**.

Chaque bloc convolutionnel contient :

- Convolution 3×3
- Batch Normalization
- ReLU
- Convolution 3×3
- ReLU
- MaxPooling

Après l’extraction des caractéristiques :

- Adaptive Average Pooling
- couche fully connected
- Dropout
- classification finale en **10 classes**

L’architecture est paramétrable avec la liste :
