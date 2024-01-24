# Instructions

![Python logo](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/110px-Python-logo-notext.svg.png 'Python')
![OpenCV logo](https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/OpenCV_Logo_with_text_svg_version.svg/122px-OpenCV_Logo_with_text_svg_version.svg.png 'OpenCV')

L'application a été conçu pour avoir le moins de dépendances à installer manuellement.

## Prérequis
Le seul prérequis pour pouvoir installer l'application est le langage Python et l'installateur Python pip.

## Installation
Pour installer le projet, il faut d'abord cloner ce [repos](https://gitlab.com/Math-Cloud/tp1-techno/) et utiliser la branche `main` ou télécharger le code source.
Ensuite, il faut lancer les scripts d'installation contenus dans le dossier `installation` selon la plateforme:
- .bat pour Windows CMD
- .ps1 pour Windows Powershell
- .sh pour Linux Bash

Le script va créer un environnement virtuel Python qui est par la suite réactivable lorsque fermé dans le dossier `./venv/bin/activate.` Encore une fois, il faut utiliser la bonne extension de script selon la plateforme. Pour désactiver l'environnement, il faut utiliser la commande `deactivate`. 

Après avoir créé et activé l'environnement, le script va installer les dépendances avec pip en utilisant le fichier requirements.txt.
Rien de plus n'est nécessaire.

## Utilisation
Une fois dans l'environnement virtuel, l'application se lance avec `python main.py` à partir de la racine du projet.
- À noter qu'ici, il faut utiliser `python` et non `python3`, car certaines librairies ne sont pas faites pour Python 3.

Plusieurs options sont disponibles dans l'application. L'utilisation de `-h` ou de `--help` retourne ceci:

| Paramètre | Description |
| ------ | ------ |
| -d, --debug | Affiche toutes les valeurs à l'écran |
| -e, --epochs=VALUE | Donne le nombre de fois que l'AI va repasser sur le _training set_ |
| -t, --training | Envoie l'AI en mode _Training_ |
| -r, --restart | Dit à l'AI de recommencer du début en regénérant des valeurs pour ses _weights_ et ses _biases_ |
| --nb_of_letters=VALUE | La valeur est le nombre de fois qu'une lettre sera répété avec des valeurs aléatoires dans le _training_set_ |

Les paramètres `epochs`, `restart` et `nb_of_letters` ne fonctionne que lorsque l'argument train est passé.

<br>

Sans paramètres, l'application ouvre une fenêtre qui va premièrement calibrer les valeurs des doigts et des mains.

Il faut tourner sa main vers l'extérieur pour un résultat optimal de calibration:<br>
![Hand](https://media.istockphoto.com/photos/womans-hand-turned-to-the-side-picture-id468356716?k=20&m=468356716&s=170667a&w=0&h=hySKeupzO5JEAl6f3hv1zywSOVDgTPpm-oA0s_FOws8= 'Hand to the side')<br>
Lorsque calibrer, il suffit maintenant de faire les signes de l'alphabet (une image avec les signes est disponible dans le dossier `doc`) et l'AI va renvoyer une valeur qui sera affichée à l'écran.

Pour l'instant, l'envoie des valeurs à l'AI ne se fait qu'avec la main droite, même si les deux sont détectées et différenciées.

## Recommendations

Les paramètres sont déjà optimisés pour donner le meilleur résultat, alors l'utilisation sans les paramètres `epochs` et `nb_of_letters` est l'idéal. Si jamais plusieurs _training_ doivent être faits, il est recommendé de garder les mêmes paramètres pour chacun pour favoriser l'apprentissage. Un changement de paramètres entre deux _trainings_ rendrait en fait les résultats pire qu'escompter.

![Mediapipe logo](https://google.github.io/mediapipe/images/logo_horizontal_color.png 'Mediapipe')
