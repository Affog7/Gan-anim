# AnimeGAN Pytorch <a href="https://colab.research.google.com/github/Affog7/Gan-anim/blob/main/notebooks/animeGAN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Dans AnimeGAN, les modèles utilisés sont des **Réseaux Antagonistes Génératifs (GANs)**, qui sont particulièrement efficaces pour les tâches de transformation d'images, comme la conversion de photos réalistes en images de style anime. Voici les détails des choix des modèles :

> **Note :** Tout le code présenté ici est exécutable sur [Google Colab](https://colab.research.google.com/github/Affog7/Gan-anim/blob/main/notebooks/animeGAN.ipynb), ce qui permet d'utiliser facilement des GPU pour accélérer l'entraînement et l'inférence.

> **NoteBook :** [Notebook](https://github.com/Affog7/Gan-anim/blob/main/notebooks/animeGAN.ipynb)

# AnimeGAN

AnimeGAN est un modèle d’apprentissage profond permettant de transformer des photos réelles en images de style anime à l’aide de réseaux antagonistes génératifs (GAN).

## Prerequisites

- Python 3.x
- Google Colab (for cloud-based execution)
- PyTorch
- CUDA (for GPU support)

## Installation

1. Clonez ce dépôt :

   ```bash
   git clone https://github.com/Affog7/Gan-anim


2. Installez les dépendances requises :

    ```bash
    pip install -r requirements.txt

3. Dataset utilisé pour l’entraînement et test :
   * https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0/dataset_v1.zip



## Explication du Choix des Modèles

Dans AnimeGAN, les modèles utilisés sont des **Réseaux Antagonistes Génératifs (GANs)**, qui sont particulièrement efficaces pour les tâches de transformation d'images, comme la conversion de photos réalistes en images de style anime. Voici les détails des choix des modèles :

### 1.  <a href="/Models_genere/GeneratorV2_train_photo_Hayao_init.pt"> Générateur (GeneratorV2)  </a>

Le générateur est responsable de créer des images de style anime à partir de photos réalistes. Plusieurs choix architecturaux ont été faits pour garantir la qualité des images générées :

- **Architecture Convolutionnelle Profonde** : Permet de capturer les détails complexes nécessaires pour la transformation des caractéristiques réalistes en style anime.
- **Residual Blocks** : Ces blocs résiduels facilitent l'apprentissage des transformations complexes tout en conservant les détails de l'image d'entrée, ce qui est crucial pour maintenir la structure de l'image originale.
- **Instance Normalization** : Cette normalisation est souvent utilisée dans les tâches de transfert de style, car elle aide à adapter efficacement le style visuel entre les images.

### 2.     <a href="/Models_genere/discriminator_train_photo_Hayao.pt"> Discriminateur (Discriminator)</a> 


Le discriminateur a pour tâche de distinguer les images générées des images réelles de style anime. Voici pourquoi certains choix ont été faits :

- **PatchGAN** : L'architecture PatchGAN évalue les images par petits patchs ou régions, ce qui permet une évaluation locale de la qualité des textures, aidant à générer des images plus détaillées et cohérentes.
- **Spectral Normalization** : Cette technique est utilisée pour stabiliser l'entraînement du discriminateur, en évitant les gradients excessifs qui pourraient déséquilibrer l'apprentissage.

### 3. Perte de GAN (GAN Loss)

Le choix de la **perte LSGAN (Least Squares GAN)**, au lieu de la perte classique de GAN, a plusieurs avantages :

- **Stabilité de l'entraînement** : La perte LSGAN atténue les problèmes de gradients instables, ce qui conduit à un entraînement plus stable et rapide.
- **Qualité des images générées** : Cette perte pousse le générateur à produire des images plus réalistes, améliorant la qualité du style anime.

### 4. Réglages des Hyperparamètres

- **Learning Rates Différents pour Générateur et Discriminateur** : Les taux d'apprentissage du générateur (`lr_g`) et du discriminateur (`lr_d`) sont ajustés différemment pour optimiser la convergence de chaque modèle.
- **Poids des différentes pertes (`wadvd`, `wadvg`, `wcon`, etc.)** : Ces hyperparamètres définissent l'importance relative des différentes composantes de la perte (adversaire, contenu, etc.), permettant un équilibre entre la qualité visuelle, la préservation du contenu et l'application du style.

En résumé, chaque choix architectural et paramétrique vise à maximiser l'efficacité de la transformation d'images réalistes en images stylisées tout en maintenant la stabilité et la qualité de l'entraînement.




## USAGE 

* Entraînement
    ```bash
    !python3 train.py --real_image_dir '/content/dataset/train_photo'\
                  --anime_image_dir '/content/dataset/Hayao'\
                  --batch 8\
                  --model v2\
                  --amp --cache\
                  --init_epochs 10\
                  --exp_dir {working_dir}\
                  --gan_loss lsgan\
                  --init_lr 0.0001\
                  --lr_g 0.00002\
                  --lr_d 0.00004\
                  --wadvd 300.0\
                  --wadvg 300.0\
                  --wcon 1.5\
                  --wgra 3.0\
                  --wcol 70.0\
                  --use_sn


* Inférence (pour transformer une image ou une vidéo) :
  
     *  Image
   
       ```bash
       python3 inference_image.py --checkpoint /path/to/model.pt\
                               --src /path/to/input/images\
                               --dest /path/to/output/images
   
   
    *  Video
   
           ```bash
           python3 inference_video.py --checkpoint /path/to/model.pt\
                                   --src /path/to/input/video.mp4\
                                   --dest /path/to/output/video.mp4

                        
## Resultats

* Image vs Images animées

    <img src="./results/image.png"/>

* Video Animée

    <a href="./Video/giphy.mp4"> Vidéo originale </a> 

    <a href="./results/test_vid_3_anime.mp4"> Vidéo animée </a>

* Modèle généré

    * <a href="/Models_genere/GeneratorV2_train_photo_Hayao.pt"> train_photo_Hayao </a> 
    
    * <a href="/Models_genere/discriminator_train_photo_Hayao.pt"> discriminator_train_photo_Hayao </a> 

    * <a href="/Models_genere/GeneratorV2_train_photo_Hayao_init.pt"> GeneratorV2_train_photo_Hayao_init </a>




