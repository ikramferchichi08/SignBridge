# SignBridge

SignBridge est un projet universitaire open source de reconnaissance en temps
réel de mots en langue des signes à partir d'une webcam ou d'une vidéo.

## Objectif

- Reconnaître des signes isolés à partir des mouvements du corps, des mains et
  du visage.
- Utiliser un modèle léger pouvant être connecté à une webcam.
- Afficher progressivement les prédictions sous forme de transcription.
- La traduction continue de phrases et la synthèse vocale sont prévues comme
  extensions, mais ne sont pas incluses dans cette phase.

## Équipe

| Étudiant | Responsabilités |
| --- | --- |
| Ikram Ferchichi | Préparation des données, extraction des landmarks, normalisation, entraînement, comparaison et MLflow |
| Malak Ghannouchi | Pipeline live, lissage temporel, DVC, API, interface, Docker et suivi du drift |
| Les deux | Évaluation, vidéos de signers personnalisées, rapport et présentation |

## Technologies utilisées

- **Python 3.11** : langage principal.
- **OpenCV** : lecture des vidéos et traitement des images.
- **MediaPipe Holistic** : détection des points du corps, des mains et du visage.
- **NumPy** : transformation, normalisation et préparation des séquences.
- **Pandas et Matplotlib** : exploration et visualisation des données.
- **PyTorch** : construction et entraînement des modèles.
- **Scikit-learn** : validation croisée par signer et calcul des métriques.
- **MLflow** : suivi des expériences et comparaison des modèles.
- **DVC** : versionnement des vidéos, landmarks et modèles.

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest
```

- MediaPipe utilisé : `0.10.21`.
- Les dépendances sont listées dans [requirements.txt](requirements.txt).

## Workflow réalisé

- **1. Téléchargement du dataset — réalisé**
  - Dataset : WLASL, un dataset de signes isolés.
  - Outils : script Python et DVC.
  - Résultat : 1 102 vidéos réparties sur 100 signes.
  - Pourquoi : disposer d'un ensemble de données contrôlé et reproductible.

- **2. Versionnement des données — réalisé**
  - Outil : DVC.
  - Données concernées : vidéos brutes, landmarks et modèles.
  - Pourquoi : éviter de stocker les gros fichiers dans Git et pouvoir
    reproduire exactement les expériences.

- **3. Exploration des métadonnées — réalisée**
  - Outils : Pandas et Matplotlib.
  - Vérifications : labels, nombre de vidéos par signe, durée, FPS, signers et
    vidéos illisibles.
  - Résultat : rapports et graphiques dans `reports/wlasl/`.
  - Pourquoi : identifier les problèmes de qualité et de déséquilibre avant
    l'entraînement.

- **4. Lecture des vidéos — réalisée**
  - Outil : OpenCV (`cv2.VideoCapture`).
  - Action : lecture image par image.
  - Optimisation : réduction des grandes images à une taille maximale de
    640 pixels.
  - Pourquoi : accélérer l'extraction sans modifier les coordonnées
    normalisées.

- **5. Extraction des landmarks — réalisée**
  - Outil : MediaPipe Holistic.
  - Points extraits : 25 points du corps, 21 points de chaque main et un
    sous-ensemble du visage composé des lèvres et des sourcils.
  - Résultat : coordonnées `x`, `y`, `z` pour chaque point.
 (x : position horizontale du point, de gauche à droite.
  y : position verticale du point, de haut en bas.
  z : profondeur approximative du point par rapport à la caméra.)
  - Pourquoi : remplacer les pixels par des caractéristiques compactes des
    mouvements et faciliter une future utilisation en temps réel.

- **6. Gestion des détections manquantes — réalisée**
  - Outil : NumPy.
  - Action : création d'un masque indiquant si le corps, chaque main ou le
    visage a été détecté dans chaque image.
  - Action : interpolation des positions entre les images disponibles.
  - Si une partie n'est jamais détectée : ses valeurs restent à zéro et son
    masque reste faux.
  - Pourquoi : limiter l'effet des pertes temporaires de détection.

- **7. Normalisation des landmarks — réalisée**
  - Outil : NumPy.
  - Centre : milieu entre l'épaule gauche et l'épaule droite.
  - Échelle : largeur entre les deux épaules.
  - Formule : `(point - centre_des_épaules) / largeur_des_épaules`.
  - Pourquoi : rendre les vidéos comparables malgré la position et la taille
    différentes des signers.

- **8. Resampling temporel — réalisé**
  - Outil : NumPy (`linspace` et interpolation).
  - Action : conversion de chaque vidéo en exactement 32 frames.
  - Pourquoi : fournir au modèle une entrée de taille fixe, quelle que soit la
    durée de la vidéo.
  - Sortie : fichiers `.npz` contenant les features et les masques de détection.

- **9. Création des features supplémentaires — réalisée**
  - Features hand-local : représentation de la forme de la main par rapport au
    poignet et à sa taille.
  - Canonicalisation : miroir des vidéos où la main gauche est dominante afin
    de standardiser la main principale.
  - Deltas : différence entre deux frames successives pour représenter le
    mouvement.
  - Masques : indication des parties réellement détectées.
  - Pourquoi : tester quelles représentations décrivent le mieux un signe.

- **10. Entraînement des modèles — réalisé**
  - Outil : PyTorch.
  - Modèles testés : GRU(), GRU bidirectionnel, Transformer et CNN-LSTM.
  (GRU : apprend l’évolution temporelle des mouvements.
  Transformer : analyse les relations entre les frames avec l’attention.
  CNN-LSTM : extrait des caractéristiques puis apprend leur évolution dans le temps.)
  - Entrée : séquences de landmarks de 32 frames.
  - Sortie : une classe parmi les 100 signes.
  - Pourquoi : comparer plusieurs architectures capables d'apprendre
    l'évolution temporelle des mouvements.

- **11. Validation par signer — réalisée**
  - Outil : `StratifiedGroupKFold` de Scikit-learn.
  - Configuration : 5 folds.
  - Règle : un même signer ne se trouve jamais à la fois dans l'entraînement et
    dans le test d'un fold.
  - Pourquoi : mesurer la généralisation à de nouvelles personnes et éviter
    une fuite de données liée au signer.

- **12. Mesure des performances — réalisée**
  - Métriques : Top-1, Top-5, Macro-F1, latence et taille du modèle.
  - Top-1 : le signe prédit en première position est correct.
  - Top-5 : le signe correct se trouve parmi les cinq meilleures prédictions.
  - Macro-F1 : moyenne équilibrée des performances sur les classes.
  - Pourquoi : comparer à la fois la précision et la possibilité d'une
    utilisation en temps réel.

- **13. Suivi des expériences — réalisé**
  - Outil : MLflow.
  - Informations enregistrées : paramètres, features, métriques par fold,
    checkpoints, taille et latence.
  - Stockage local : dossier `mlruns/`.
  - Pourquoi : conserver une trace reproductible de chaque entraînement.

- **14. Comparaison des modèles — réalisée**
  - Script : `src.compare_runs`.
  - Fichiers générés : `reports/model_comparison.csv` et
    `reports/model_comparison.md`.
  - Pourquoi : sélectionner le meilleur compromis entre précision, taille et
    rapidité.

- **15. Enregistrement de clips personnalisés — réalisé**
  - Outil : OpenCV.
  - Script : `src/record_clips.py`.
  - Action : enregistrer une vidéo à la fois pour un signe et un signer précis.
  - Organisation : `data/custom_clips/<signe>/<signer>_<numero>.mp4`.
  - Pourquoi : tester le modèle sur les propres gestes des membres de l'équipe,
    sans utiliser ces clips pendant l'entraînement.

- **16. Évaluation sur des signers personnalisés — réalisée**
  - Outil : MediaPipe, PyTorch et Scikit-learn.
  - Script : `src/eval_custom_clips.py`.
  - Étapes : extraction des landmarks, découpage autour des mains,
    resampling à 32 frames, prédiction et calcul des métriques.
  - Résultats générés : `reports/custom_eval/predictions.csv`,
    `summary.json` et `custom_features_T32.npz`.
  - Pourquoi : mesurer la généralisation sur de nouvelles personnes et conserver
    les features pour un futur suivi du drift.

- **17. Démo live des signes — réalisée**
  - Outil : OpenCV, MediaPipe, PyTorch et `WordSmoother`.
  - Script : `src/live_demo.py`.
  - Fonctionnement : lecture webcam ou vidéo, buffer temporel, prédictions
    régulières, lissage, seuil de confiance, cooldown et transcription.
  - Sortie : mot reconnu, confiance, FPS et liste des mots acceptés.
  - Pourquoi : fournir une première démonstration interactive proche du cas
    d'utilisation réel.

- **18. Interface React + FastAPI — réalisée**
  - Frontend : React, Vite, Recharts, Lucide.
  - Backend : FastAPI.
  - Pages : vue d'ensemble, exploration du dataset, comparaison des expériences
    et état de la démo live.
  - API : `/api/health`, `/api/overview`, `/api/dataset` et `/api/experiments`.
  - Pourquoi : présenter les données et les résultats du projet dans une
    interface interactive lisible.

## Résultats actuels

- Dataset : **1 102 vidéos** et **100 signes**.
- Séquences extraites : **1 102 séquences**, chacune avec **32 frames**.
- Évaluation : **5 folds** séparés par signer.
- Meilleur modèle : **Transformer avec features hand-local**.
- Top-1 : **31,5 % ± 8,6 %**.
- Top-5 : **57,2 %**.
- Macro-F1 : **0,302 ± 0,084**.
- Paramètres : **348 772**.
- Taille : **1,41 MB**.
- Baseline GRU avec landmarks bruts : **18,1 % Top-1** et **0,171 Macro-F1**.
- Conclusion : les features hand-local améliorent la reconnaissance par rapport
  aux landmarks bruts tout en conservant un modèle léger.

Résultats détaillés :

- [reports/model_comparison.md](reports/model_comparison.md)
- [reports/model_comparison.csv](reports/model_comparison.csv)

## Vérification et MLflow

Interface React + FastAPI :

```powershell
python -m pip install -r requirements.txt
uvicorn app.main:app --reload

cd app/frontend
npm install
npm run dev
```

Pages disponibles :

- Vue d'ensemble : statistiques du projet et workflow réalisé.
- Dataset : recherche des signes, graphiques, métadonnées et benchmark.
- Expériences : comparaison interactive des modèles, Top-1, Top-5, Macro-F1
  et taille des modèles.
- Démo live : état du checkpoint final et instructions d'inférence.
- Comptes : inscription et connexion par e-mail, avec historique privé par
  utilisateur ; le compte administrateur peut consulter l'historique de tous les
  utilisateurs.

## Docker

L'image Docker sert le frontend compilé et l'API FastAPI sur le même port.
Les données, rapports et modèles restent montés en lecture seule depuis le
workspace ; la base SQLite utilise un volume Docker persistant. Avant le
démarrage, récupérez les données DVC si elles ne sont pas déjà présentes :

```powershell
dvc pull
```

Le compte administrateur est facultatif. Pour le créer au premier lancement,
définissez ses identifiants dans PowerShell puis démarrez Compose :

```powershell
$env:SIGNBRIDGE_ADMIN_EMAIL = "admin@example.com"
$env:SIGNBRIDGE_ADMIN_PASSWORD = "choose-a-private-password-of-12-chars-or-more"
docker compose up --build -d
```

Sans compte administrateur, laissez ces variables non définies et lancez
directement `docker compose up --build -d`. Ouvrez ensuite
<http://localhost:8000>. Consultez l'état avec `docker compose ps`, les logs
avec `docker compose logs -f signbridge`, et arrêtez les conteneurs avec
`docker compose down` (le volume de l'historique reste conservé).

La démo webcam reste à lancer sur l'hôte Windows, car Docker Desktop ne reçoit
pas automatiquement l'accès à la webcam. Elle peut envoyer les signes reconnus
vers l'API Docker avec `python -m src.live_demo --record-history
--api-url http://127.0.0.1:8000`.

En exécution locale, les comptes et événements sont conservés dans
`data/signbridge.sqlite3` ; sous Docker, ils persistent dans le volume
`signbridge-state`.
Les mots de passe sont stockés sous forme de dérivés PBKDF2, et les sessions
utilisent un cookie `HttpOnly`. Pour créer un administrateur au premier
démarrage, configurez ses identifiants avant de lancer l'API :

```powershell
$env:SIGNBRIDGE_ADMIN_EMAIL = "admin@example.com"
$env:SIGNBRIDGE_ADMIN_PASSWORD = "change-this-to-a-private-password"
uvicorn app.main:app --reload
```

Le mot de passe administrateur doit contenir au moins 12 caractères. Ces
variables ne sont pas nécessaires pour les comptes normaux. En production,
servez le site en HTTPS et définissez `SIGNBRIDGE_COOKIE_SECURE=true`.

Pour enregistrer les signes reconnus et les événements de connexion dans
l'historique du compte, lancez la démo live depuis le terminal avec
`--record-history`. Elle demande les identifiants du compte sans afficher le
mot de passe :

```powershell
python -m src.live_demo --checkpoint models\transformer_final\final.pt --record-history
```

Test avec une vidéo :

```powershell
python -m src.test_setup --video path\to\video.mp4 --no-display
```

Test avec une webcam :

```powershell
python -m src.test_setup
```

- Appuyer sur `q` pour arrêter la fenêtre webcam.
- Le script affiche les versions des packages, les détections et le FPS.
- Il crée également un run MLflow de vérification.

Interface MLflow locale :

```powershell
mlflow ui --backend-store-uri .\mlruns
```

## Structure du projet

```text
data/{raw,landmarks,processed}/  Données gérées par DVC
src/                             Scripts Python
notebooks/                       Exploration et essais
configs/                         Configuration YAML
app/                             API et interface prévues
models/                          Modèles gérés par DVC
reports/                         Résultats et graphiques
tests/                           Tests automatisés
docs/                            Documentation et problèmes
```

## Prochaines étapes

- **Application live web — amélioration prévue**
  - démarrer la webcam directement depuis l'interface React ;
  - transmettre les frames ou les landmarks avec WebSocket ;
  - afficher la transcription dans le navigateur ;
  - ajouter le choix du checkpoint final ;
  - conserver l'historique des prédictions ;
  - surveiller le drift des données ;
  - finaliser le packaging Docker.

## Limites

- Le système reconnaît des mots isolés et ne réalise pas encore une traduction
  continue.
- Le système est un assistant et ne remplace pas un interprète qualifié.
- Les données WLASL sont utilisées dans un cadre académique et informatique ;
  elles ne doivent pas être utilisées commercialement sans vérifier leur licence.
