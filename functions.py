# Fonctions récurrentes lors de la réalisation de projets.
# Nous les stockons ici pour qu'elles soient facilement accessibles.

import datetime
import itertools
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


# Créer une fonction pour importer une image et la redimensionner pour pouvoir l'utiliser avec notre modèle.


def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Lit une image à partir du nom du fichier, la transforme en un tenseur et la remodèle en
    (224, 224, 3).

    Paramètres
    ----------
    filename (str) : nom de l'image cible (chaîne de caractères)
    img_shape (int) : taille à laquelle redimensionner l'image cible, par défaut 224
    scale (bool) : mise à l'échelle des valeurs de pixels dans l'intervalle(0, 1), par défaut True
    """
    # Lire dans l'image
    img = tf.io.read_file(filename)
    # Le décoder en un tenseur
    img = tf.image.decode_jpeg(img)
    # Redimensionner l'image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Redimensionner l'image (obtenir toutes les valeurs entre 0 et 1)
        return img / 255.
    else:
        return img


# Remarque : Le code de la matrice de confusion qui suit est un remix de la matrice de Scikit-Learn.
# Fonction plot_confusion_matrix :
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html

# Notre fonction doit avoir un nom différent de celui de la fonction plot_confusion_matrix de sklearn.
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    """
    Crée une matrice de confusion étiquetée comparant les prédictions et les étiquettes du ground truth.

    Si les classes sont passées, la matrice de confusion sera étiquetée, sinon, les valeurs entières de classe
    seront utilisées.

    Arguments :
      y_true : Tableau d'étiquettes de vérité (doit avoir la même forme que y_pred).
      y_pred : Tableau d'étiquettes prédites (doit avoir la même forme que y_true).
      classes : Tableau d'étiquettes de classe (par exemple, sous forme de chaîne). Si `None`,
       des étiquettes entières sont utilisées.
      figsize : Taille de la figure de sortie (default=(10, 10)).
      text_size : Taille du texte de la figure de sortie (default=15).
      norm : normaliser les valeurs ou non (default=False).
      savefig : sauvegarde la matrice de confusion dans un fichier (default=False).

    Retourne :
      Un graphique de matrice de confusion étiqueté comparant y_true et y_pred.

    Exemple d'utilisation :
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # étiquettes prédites
                            classes=class_names, # tableau de noms d'étiquettes de classe
                            figsize=(15, 15),
                            text_size=10)
    """
    # Créer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / \
              cm.sum(axis=1)[:, np.newaxis]  # normalisation
    # trouver le nombre de classes avec lesquelles nous travaillons
    n_classes = cm.shape[0]

    # Tracer la figure et la rendre élégante.
    fig, ax = plt.subplots(figsize=figsize)
    # Les couleurs représentent le degré de justesse d'une classe, plus sombre == meilleur.
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Y a-t-il une liste de classes ?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Axes des étiquettes
    ax.set(title="Matrice de confusion",
           xlabel="Étiquette prédite",
           ylabel="Véritable étiquette",
           # créer suffisamment de créneaux d'axe pour chaque classe
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           # les axes seront étiquetés avec des noms de classe (s'ils existent) ou des ints.
           xticklabels=labels,
           yticklabels=labels)

    # Faire apparaître les étiquettes de l'axe des x en bas
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Définition du seuil pour les différentes couleurs
    threshold = (cm.max() + cm.min()) / 2.

    # Trace le texte sur chaque cellule
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Enregistre la figure dans le répertoire de travail actuel
    if savefig:
        fig.savefig("confusion_matrix.png")


# Créer une fonction de prédiction sur les images et les tracer (fonctionne avec les classes multiples).


def pred_and_plot(model, filename, class_names):
    """
    Importe une image située au nom de fichier, fait une prédiction sur celle-ci avec un modèle entraîné
     et affiche l'image avec la classe prédite comme titre.
    """
    # Importer l'image cible et la prétraiter
    img = load_and_prep_image(filename)

    # Faire une prédiction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Obtenir la classe prédite
    if len(pred[0]) > 1:  # vérification de la présence de classes multiples
        # si plus d'une sortie, prendre le maximum
        pred_class = class_names[pred.argmax()]
    else:
        # si une seule sortie, arrondir
        pred_class = class_names[int(tf.round(pred)[0][0])]

    # Tracer l'image et la classe prédite
    plt.imshow(img)
    plt.title(f"Prédiction: {pred_class}")
    plt.axis(False)


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Crée un callback TensorBoard à la place pour stocker les fichiers journaux.

    Stocke les fichiers journaux avec le chemin de fichier :
      "nom_du_repertoire/nom_de_l'expérience/heure_actuelle/"

    Arguments :
      dir_name : répertoire cible pour stocker les fichiers journaux de TensorBoard
      experiment_name : nom du répertoire de l'expérience (par exemple, net_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + \
              datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Sauvegarde des fichiers journaux de TensorBoard vers : {log_dir}")
    return tensorboard_callback


# Tracez les données de validation et d'entraînement séparément


def plot_loss_curves(history):
    """
    Renvoie des courbes de perte distinctes pour les métriques d'entraînement et de validation.

    Arguments :
      historique : Objet historique du modèle TensorFlow
      (voir : https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Graphique de perte
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Perte')
    plt.xlabel('Époques')
    plt.legend()

    # Graphique d'accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Époques')
    plt.legend()


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compare deux objets historiques de modèle TensorFlow.

    Arguments :
      original_history : Objet historique du modèle original (avant new_history)
      new_history : Objet historique provenant de l'entraînement continue du modèle (après original_history)
      initial_epochs : Nombre d'époques dans original_history (le tracé de new_history commence ici) 
    """

    # Obtenir les mesures originales de l'historique
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combiner l'historique originale avec la nouvelle historique
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Faire des tracés
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Accuracy de l\'entraînement')
    plt.plot(total_val_acc, label='Accuracy de la validation')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Commencer le tuning fin')  # décalage du tracé autour des époques
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Perte d\'entraînement')
    plt.plot(total_val_loss, label='Perte de validation')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Commencer le tuning fin')  # décalage du tracé autour des époques
    plt.legend(loc='upper right')
    plt.title('Perte d\'entraînement et de validation')
    plt.xlabel('époque')
    plt.show()


# Créer une fonction pour décompresser un fichier zip dans le répertoire de travail actuel.


def unzip_data(filename):
    """
    Dézippe le nom du fichier dans le répertoire de travail actuel.

    Arguments :
      filename (str) : un chemin d'accès à un dossier zip cible à dézipper.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()


# Parcours un répertoire de classification d'images
# et découvre combien de fichiers (images) se trouvent dans chaque sous-répertoire.


def walk_through_dir(dir_path):
    """
    Parcourt le répertoire en retournant son contenu.

    Arguments :
      dir_path (str) : répertoire cible

    Retourne :
      Un affichage de :
        nombre de sous-répertoires dans dir_path
        nombre d'images (fichiers) dans chaque sous-répertoire
        le nom de chaque sous-répertoire
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"Il y a {len(dirnames)} dossiers et {len(filenames)} images dans '{dirpath}'.")


# Fonction pour évaluer: accuracy, precision, recall, f1-score


def calculate_results(y_true, y_pred):
    """
    Calcule : accuracy, precision, recall et f1 score d'un modèle de classification binaire.

    Arguments :
        y_true : étiquettes vraies sous la forme d'un array 1D
        y_pred : étiquettes prédites sous la forme d'un array 1D

    Retourne un dictionnaire avec : accuracy, precision, recall, f1-score.
    """
    # Calculer l'accuracy du modèle
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculer : precision, recall et f1 score du modèle en utilisant la "moyenne pondérée".
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results
