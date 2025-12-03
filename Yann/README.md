# Projet Dynamic Pricing — OptiZoignons (Yann)

✅ Ce dépôt contient du code, des notebooks et des données pour des exercices et simulations sur la tarification dynamique (duopoly / compétition à deux vendeurs). L'objectif principal est d'explorer des stratégies de pricing, des simulations de compétition et des approches de dynamic programming pour maximiser le revenu sous contraintes de capacité.

---

## Structure du dépôt

Voici les fichiers et dossiers les plus importants retrouvés à la racine et dans le workspace :

- `duopoly.py` — exemple d'algorithme de tarification (fonction p(...)) attendu par l'environnement de simulation. L'algorithme reçoit l'historique des prix/demandes et renvoie un prix et un `information_dump` (stateful).
- `tools/run_a_simu.py` — utilitaire pour lancer une simulation locale pas-à-pas en utilisant `duopoly.p` et les fichiers CSV de `duopoly_competition_details.csv`.
- `tools/trace_cap_util.py` — utilitaire de visualisation pour tracer les courbes d'utilisation de capacité.
- `requirements.txt` — liste complète des dépendances Python.
- `environment.yml` — environnement conda (nommé `opti_env`) pour recréer l'environnement complet (y compris la partie conda/pip listée).
- `test.py` — fichier de tests / utilitaire (actuellement vide dans la version fournie).
- `week_n/W8/W8_20241112_DynamicProgramming.ipynb` — notebook sur le dynamic programming appliqué au pricing (exercices, implémentations et exemples).
- `Quentin_data/` et `Results/` — dossiers contenant des données d'exemple, détails de compétition, résultats et CSV utilisés par les scripts et notebooks.

---

## Buts / cas d'usage

Ce repository sert principalement à :

- Expérimenter des stratégies de tarification pour une compétition en duopole.
- Lancer des simulations (avec `tools/run_a_simu.py`) pour tester des comportements sous scénarios issus de CSVs.
- Étudier des méthodes de dynamic programming pour définir une stratégie optimale de prix dans le temps (voir notebooks W8 et autres notebooks de la série `week_n`).

---

## Installation (recommandée — conda)

Si vous utilisez Anaconda/Miniconda (recommandé), recréez l'environnement exactement comme fourni :

PowerShell (Windows):

```powershell
# depuis le dossier du repo
conda env create -f environment.yml
conda activate opti_env
```

Puis vérifiez que Python et les paquets nécessaires sont disponibles.

Alternativement, pour un venv pip-only (si vous n'utilisez pas conda) :

```powershell
python -m venv env
env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note : L'environnement `environment.yml` est très complet (packages de recherche, optimisation, TensorFlow, PyTorch, etc.). Vous pouvez installer une sous-sélection de dépendances si vous n'avez besoin que de l'analyse / visualisation (numpy, pandas, matplotlib, tqdm, jupyter).

---

## Comment lancer une simulation locale (exemples)

Les simulations utilisent `duopoly.p` (l'algorithme participant). `tools/run_a_simu.py` montre un exemple d'utilisation au fil du temps à partir des fichiers `duopoly_competition_details.csv`.

Exécuter la simulation interactive (depuis la racine du repo) :

```powershell
# activer l'env conda
conda activate opti_env
# lancer Python interactif ou exécuter le script dans un notebook
python -c "from tools import run_a_simu; run_a_simu.run_a_simu('duopoly_competition_details.csv', s=1, max_t=20)"
```

Ou ouvrir le notebook `tools/run_a_simu.py` et appeler `run_a_simu(...)` depuis un notebook pour obtenir l'historique `information_dump` et les facteurs calculés.

Après une simulation, `duopoly_feedback.data` est sauvegardé à la fin d'une saison (si `day >= 100` dans l'exemple) — c'est un pickle contenant l'historique et l'état. Certains scripts locaux cherchent aussi `target_sales_curve_quentin.pkl`.

---

## Notebooks & exercices

- `week_n/W8/W8_20241112_DynamicProgramming.ipynb` — implémente et explique des techniques de dynamic programming appliquées au pricing (calcul analytique du prix optimal, simulation par pas, matrice de fonction de valeur, etc.).
- D'autres notebooks dans `week_n/` et `Quentin_data/` contiennent analyses additionnelles, modèles OLS pour la demande, et jeux de données d'exemple.

Conseil : ouvrez ces notebooks avec Jupyter / JupyterLab après activation de l'environnement conda.

---

## Fichier `duopoly.py` — contract de l'algorithme

La fonction principale `p(...)` doit respecter ce contrat :

- Entrées principales :
  - `current_selling_season` (int)
  - `selling_period_in_current_season` (int)
  - `prices_historical_in_current_season` (np.ndarray ou None) — historique prix (own/competitor)
  - `demand_historical_in_current_season` (np.ndarray ou None)
  - `competitor_has_capacity_current_period_in_current_season` (bool)
  - `information_dump` (objet réutilisable pour l'état interne de l'algorithme)

- Retour : typiquement `(price_today, information_dump, demand)` ou `(price_today, information_dump)` selon implémentation.

L'exemple `duopoly.py` fourni initialise le `information_dump` au premier jour, sauvegarde l'historique dans un pickle `duopoly_feedback.data`, et applique des règles simples (ex. recalcul tous les 5 jours, prix plancher).

---

## Visualisation & outils auxiliaires

Utilisez `tools/trace_cap_util.py` pour tracer rapidement les courbes d'utilisation de capacité après une simulation.

---

## Tests et développement

- `test.py` est présent mais vide — vous pouvez ajouter vos tests unitaires ou scripts d'expérimentation.
- Pour automatisation : considérer `pytest` et ajouter un `tests/` avec cas simples sur `duopoly.p`.

---

## Conventions et recommandations

- Lancer les notebooks depuis la racine du projet après activation de l'environnement conda.
- Mettre à jour `duopoly.p` (ou créer d'autres modules) pour expérimenter différentes stratégies.

---

## Données & résultats

- `Quentin_data/` — exemples d'entrée (CSV, notebooks) brûches d'analyse.
- `Results/` — répertoires datés avec sorties de simulations, CSV et analyses (utilisez-les comme dataset/benchmarks pour vos algorithmes).

---

## Contribuer / contact

Si tu veux que j'ajoute :
- des scripts d'exécution plus robustes (CLI),
- des tests unitaires et CI,
- ou un petit guide de contribution (CONTRIBUTING.md),

je peux le faire — dis-moi ce que tu veux prioriser.

---

Licence

Ce dépôt ne contient pas de fichier de licence explicite — si tu veux en ajouter une (MIT, Apache-2.0, etc.), je peux l'ajouter.

Bonne exploration ! 🚀
