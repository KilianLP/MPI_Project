# Projet : Machine Learning Distribué avec MPI

Ce dépôt contient le code du projet Machine Learning distribué avec MPI.

## Détails techniques

La majorité des détails techniques sont décrits dans le compte rendu technique du projet.
Veuillez vous référer à ce document pour une description approfondie de l’architecture, des choix méthodologiques et des résultats expérimentaux.

## Installation

Ce projet est développé en Python.
Pour assurer le bon fonctionnement du code, il est recommandé de créer un environnement virtuel, soit avec Miniconda, soit avec venv, puis d’installer les dépendances requises :

pip install -r requirements.txt

## Lancement des scripts

Pour lancer l’entraînement, placez-vous à la racine du projet et exécutez la commande suivante :

mpirun -np 4 python3 -m train.gpt_mpi \
  --file data/tinyshakespeare.txt \
  --epochs 1 \
  --batch-size 32 \
  --block-size 128 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 128 \
  --lr 3e-4

## Paramètres

Les paramètres utilisés sont les suivants :

- --file : chemin vers le jeu de données textuel utilisé pour l’entraînement.
- --epochs : nombre d’époques d’entraînement.
- --batch-size : taille des mini-batchs traités par chaque processus MPI.
- --block-size : longueur maximale des séquences d’entrée.
- --n-layer : nombre de couches du modèle Transformer.
- --n-head : nombre de têtes d’attention par couche.
- --n-embd : dimension des embeddings.
- --lr : taux d’apprentissage utilisé par l’optimiseur.

L’option -np 4 permet de spécifier le nombre de processus MPI utilisés pour l’entraînement.
Dans cet exemple, l’entraînement est réparti sur quatre processus parallèles.

La durée de l’entraînement peut varier en fonction du nombre de processus, de la taille du modèle et du jeu de données utilisé.



Kilian Preuss