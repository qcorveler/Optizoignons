# Optizoignons

## Lien canva
https://www.canva.com/design/DAG2t1rRtMg/Hw2CsxX2Y7zGM7q9hEepUw/edit?utm_content=DAG2t1rRtMg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

---
---

## Organisation du git

### └ Travail commun
Dans ce dossier on trouve les résultats des notebooks d'exercices fait en cours et d'autres ressources fournies par les profs.

### └ Travail perso
Dans ce dossier on trouve un sous-dossier par personne pour faire un peu bac à sable et tester ce qu'on veut avec les ressources qu'on veut.

### └ Resultats
C'est là qu'on met tous nos résultats, organisés de la manière suivante : \
└ Un sous-dossier par jours où on récupère les résultats \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└ Un sous-sous-dossier par personne
 
C'est aussi là que je propose de mettre le fichier d'analyse des résultats et de mettre les courbes qu'on peut en tirer, dans un sous-dossier Analyse en essayant de le ranger au mieux possible

### └ Duopoly
Ici on a un sous-dossier par personne pour juste mettre notre duopoly le plus récent
(On peut imaginer avoir un sous-sous-dossier par jour pour avoir une trace des anciens duopoly)

---
---


# TODO
On pourrait apprendre 3 ou 4 scénarios de demande différente, genre low demand, medium, high ... Essayer de déterminer à l'avance dans quel cas on est, et en fonction ajuster différemment le prix


Pour la présentation :
- Faire 3 ou 4 scénarios de demande différentes (low, medium, high ...)
- Essayer de faire une estimation dynamique du prix du compétiteur (par exemple récupérer les données pendant 20 saisons et les utiliser pour les 80 suivantes)


---
---

## Stratégies utilisées

#### Random
Juste mettre des prix randoms, ça nous a donné une première idée de la courbe de demande

#### Laisser son adversaire finir son stock très tôt pour être seul sur le marché à la fin
Ça a pas super bien marché

#### Suivre la courbe idéale de vente
On adapte le prix pour essayer de coller nos ventes sur le niveau de vente qui nous donnait de bons résultats initialement (on augmente le prix si on vend trop par rapport à l'historique et on le baisse si on vend pas assez)

## Idées de statégies à implémenter pour la suite

#### Faire une régression linéaire sur la courbe d'utilisation de la capacité
essayer de modéliser la courbe de la demande avec une regression linéaire (ou potentiellement x régressions linaires sur x tronçons de la courbe) pour prédire 

## Questions à poser au prof

Demander la différence entre booking et capacity curve, 

réexpliquer sur quoi faire la regression linéaire et à quoi elle sert.

Comment avancer ?

# Présentation # 1

Plan :
- Introduction
- Présentation du dynamic pricing
- Présentation de la compétition
- Première stratégie -> RANDOM (premiers résultats)
- Estimation de la booking curve
- Implémentation d'un algo pour suivre la booking curve "optimale"
- Estimation de la demande en fonction du temps -> Observation de la tendance (ça augmente avec le temps)
- Estimation et du revenu en fonction de notre prix et du prix du compétiteur (en 5 fois 20 jours)
- Parler de comment implémenter un algo qui prenne en compte tout ça
- Améliorations possibles 
	+ Calcul de la sensibilité
	+ Calcul de l'Elasticité 
  


# Cours du 11.11.25
Les régressions linéaires vues pendant les présentations étaient nulles
-> On va essayer de comprendre pourquoi

Regression model \
*d(p1, p2) = beta0 + beta1\*p1 + beta2\*p2*

The real demand values are integers but our demand function returns a float.

Objective functions pour la régression :
- Somme du carré des différences (OLS)
- Somme de la valeur abs des différences divisées par le nombre d'observation (LAD)

Assumptions : 
- d décroît de façon monotone avec p1 (notre prix)
- d croît de façon monotone avec p2 (prix du concurrent)
- d dépend un peu du temps (croît de façon monotone)

=> *d(p1, p2, t) = beta_0t + beta_1t\*p1 + beta_2t\*p2*

avec pas beaucoup de différence entre t-1, t et t+1 pour les paramètre : \
*|beta_kt - beta_kt-1| <= delta*

- d est symétrique selon p1 et p2

*d(p1, p2, t) = d_own* \
*d(p2, p1, t) = d_comp*

Donc maintenant, on doit intégrer le temps, les valeurs entières et des paramètres légèrement différents en adéquation avec le temps *t*.

En utilisant une fonction pour arrondir à l'entier le plus proche on va déjà peut être améliorer nos modèles

ce qui nous amène à l'introduction de pyomo (bibliothèque python)

### Pyomo introduction

mobook.github.io/MO-book/intro.html

pyomo.readthedocs.io/en/6.8.0/tutorial_examples.html


