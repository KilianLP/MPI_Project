# Projet : Machine Learning Distribué avec MPI

Ce dépôt contient le code du projet **Machine Learning distribué avec MPI**.

## Détails techniques

La majorité des détails techniques sont décrits dans le **compte rendu technique** du projet.  
Veuillez vous référer à ce document pour une description approfondie de l’architecture, des choix méthodologiques et des résultats expérimentaux.

## Commentaire

Toutes les lignes de code utilisant MPI sont annotées avec un court commentaire au-dessus (`# i used MPI here`), afin de faciliter la compréhension de l’utilisation de MPI dans ce projet.


## Installation

Ce projet est développé en **Python**.  
Pour assurer le bon fonctionnement du code, il est recommandé de créer un environnement virtuel, soit avec **Miniconda**, soit avec **venv**, puis d’installer les dépendances requises :

pip install -r requirements.txt

## Lancement des scripts

Pour lancer l’entraînement avec MPI, placez-vous à la racine du projet et exécutez la commande suivante :

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

L’option `-np 4` permet de spécifier le nombre de processus MPI utilisés pour l’entraînement.  
Dans cet exemple, l’entraînement est réparti sur **quatre processus parallèles**.

La durée de l’entraînement peut varier en fonction du nombre de processus, de la taille du modèle et du jeu de données utilisé.  
Dans notre cas, sur une machine personnelle, l’entraînement a pris **environ 2 heures pour une époque**.

## Entraînement avec FSDP

Pour lancer l’entraînement en utilisant l’algorithme **FSDP (Fully Sharded Data Parallel)**, utilisez la commande suivante :

mpirun -np 4 python3 -m train.gpt_fsdp_manual \
  --file data/tinyshakespeare.txt \
  --epochs 2 \
  --batch-size 32 \
  --block-size 128 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 128 \
  --lr 3e-4

## Génération de texte

Une fois le modèle entraîné, vous pouvez générer du texte à partir d’un checkpoint en utilisant la commande suivante :

python3 -m train.gpt_generate \
  --file data/tinyshakespeare.txt \
  --checkpoint checkpoints/gpt_checkpoint.pt \
  --prompt "To be, or not to be" \
  --max-new-tokens 200 \
  --temperature 0.9 \
  --top-k 40

Le prompt peut être modifié librement.  
Il est toutefois recommandé d’utiliser un prompt **en anglais**, le modèle ayant été entraîné sur un corpus anglophone.

---

Kilian Preuss
