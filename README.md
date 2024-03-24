Prediction de la Duree de Vie Residuelle de la Batterie: 
La duree de vie d’une batterie est un facteur determinant dans la performance globale d’un systeme, influencant sa fiabilite,
son cout operationnel et son impact environnemental. La prediction de la duree de vie residuelle vise a anticiper le moment ou
une batterie atteindra la fin de sa vie utile, permettant ainsi une planification proactive de la maintenance et une optimisation de
l’utilisation des ressources.

Ce projet se concentre sur l’application de techniques avancees de machine learning pour predire la duree de vie residuelle
des batteries lithium-ion. L’objectif est de developper un modele de machine learning capable d’estimer la duree de vie
residuelle (Remaining Useful Life - RUL) des batteries lithium-ion. Cette tache est cruciale pour optimiser la gestion des
batteries, prolonger leur duree de vie et garantir leur performance dans diverses applications, allant des vehicules electriques
aux systemes de stockage d’energie. 

Les objectifs principaux de ce projet sont les suivants :
— Concevoir et mettre en œuvre un modele de machine learning capable de predire la duree de vie residuelle des batteries
lithium-ion.
— Utiliser le jeu de donnees fourni pour entraıner, tester et evaluer le modele.
— Analyser les resultats obtenus et fournir des recommandations pour l’amelioration de la gestion des batteries

Ensemble de Donnees
Le jeu de donnees fourni contient des informations sur les cycles de vie de batteries lithium-ion. Chaque observation du
dataset comprend plusieurs parametres, tels que les mesures de tension, de temperature, de courant, etc. associees a chaque
cycle de decharge/charge de la batterie. De plus, le dataset inclut la colonne RUL (Remaining Useful Life), qui represente la
duree de vie residuelle de la batterie pour chaque cycle. 

Vous retrouverez l'ensemble du code final dans le dossier "RenduFinal"
Ce dossier contient un fichier codepy\main.py qui est le seul fichier a run.
Les autres .py sont les différents fichiers crées pour faire les functions des différentes parties du code incluant:
- Le nettoyage des données
- Le feature eng qui comprends une analyse des données corrigées afin de trouver et supprimer les correlations entre paramètre
- Les 3 modèles testés (Regression Linéaire, SVM et ProcessGaussien)
- L'analyse des résultats avec le calcul d'un coeficient de performance

L'ensemble des décisions de run des fichiers se fait en modifiant les paramètres a et b --> main(a,b)
Vous pouvez aussi modifié la valeur du paramètre priority_result entre 0 et 100 qui définit l'importance de la précision VS le temps d'entrainement pour le calcul du coeficient de performance

Le dossier data contient les fichiers CSV suivants:
- Base de données initiale (Battery_RUL)
- Base de données nettoyée (Battery_RUL_clean_data)
- Base de donnée après feature eng (Battery_RUL_usable_data)
- Tableau des résultats (Battery_RUL_result)

Enfin  le dossier Doc repertorie l'ensemble des documents comme le sujet, des captures d'écran des courbes etc