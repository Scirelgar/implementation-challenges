# Challenges d'implémentation

Ce répertoire contient des énoncés de défis d'implémentation pour tester vos compétences en développement d'algorithmes quantiques.

## Structure du projet

- `/challenges` : Contient les énoncés des défis dans un format Markdown. Chaque énoncé contient une description du défi, des instructions pour l'implémentation et des suggestions de résultats à produire.
- `/notebooks` : Contient des notebooks Jupyter servant à faciliter la compréhension et le développement des solutions aux défis. On y trouve des exemples de code qui peuvent être tirés des énoncés ou qui complémentent ces derniers. L'intention est de fournir un support, particulièrement pour les notions qui ne relèvent pas directement de la **programmation** quantique mais qui sont nécessaires pour résoudre les défis.
- `/solutions` : Contient seulement un template architectural pour les solutions aux défis. Les solutions ne doivent pas être développées directement dans ce répertoire. Elle doivent être développées dans une fork ou un branche dédiée. Une fois la solution développée, elle peut être mergée via une pull request sur la branche dédiée `solutions`.

> [!IMPORTANT]
> <table>
>   <tr>
>     <td>
>       Ce répertoire ne contient pas de solution, vous pouvez cependant consulter les solutions des membres qui ont "fork" le répertoire.
>     </td>
>     <td>
>       <img width="250" alt="Exemple de fork" src="https://github.com/user-attachments/assets/67ebb506-a98e-45f5-8438-04b5dd74160b" />
>     </td>
>   </tr>
> </table>


## Premiers pas

Pour guider votre travail, il est recommandé de procéder comme suit :

1. Créez une fork du projet et clonez-le sur votre machine locale.
2. Identifiez le défi que vous souhaitez résoudre dans le répertoire `/challenges` et installez les dépendances nécessaires pour ce défi en utilisant `uv sync --package <nom_du_défi>`.

> [!TIP]
> Les noms des collections de dépendances sont définis dans les fichiers `pyproject.toml` de chaque défi. Vous pouvez consulter ces fichiers pour connaître la valeur à utiliser pour `<nom_du_défi>`.

3. Lisez attentivement l'énoncé du défi, les étapes suggérées et les résultats attendus.
4. Parcourez le ou les notebooks Jupyter associés au défi pour mieux comprendre les concepts prérequis et les exemples de code.
5. Implémentez votre solution !

## Soumission d'un nouveau défi dans le répertoire principal

Si vous souhaitez soumettre un nouveau défi, vous devez créer une pull request depuis votre fork ou votre branche dédiée vers la branche `challenges` du projet principal. Un défi doit contenir au minimum un énoncé dans un fichier markdown et un fichier `pyproject.toml` pour définir les dépendances nécessaires à l'implémentation de la solution. Il est également recommandé d'ajouter un notebook Jupyter dans le dossier `notebooks/` pour faciliter la compréhension du défi et le développement de la solution.

L'énoncé du défi doit au minimum contenir les sections suivantes :

- Objectif : une description claire et concise de ce que le défi cherche à accomplir.
- Données fournies (si applicable) : une description des données fournies pour le défi, y compris leur format et leur structure. Des instructions sur la manière d'accéder à ces données peuvent également être incluses.
- Critères de réussite : une description des critères permettant de valider si le défi a été résolu avec succès. À quoi s'attend-on comme sortie ou résultat final ?

D'autres idées de section pertinentes à inclure dans l'énoncé du défi :

- Des pistes pour l'implémentation : identifiez des pistes pour le développement de la solution et proposez un plan d'action pour guider le développement de la solution.
- Une description mathématique du problème : une définition d'une fonction de coût, l'expression d'un Hamiltonien, représentation d'un circuit quantique, etc.
- Résultats suggérés à produire : des suggestions de résultats à produire pour démontrer que les critères de réussite ont été atteints. Il peut s'agir de graphiques, de tableaux, d'analyses statistiques, etc.
- Des ressources supplémentaires : des liens vers des ressources supplémentaires pour aider à la compréhension du défi et au développement de la solution. Il peut s'agir de tutoriels, d'articles de recherche, de vidéos, etc.

## Configuration de l'environnement de développement

> [!IMPORTANT]
> La configuration actuelle est prévue pour `uv`. Utilisez `uv sync` pour installer les dépendances et `uv run` pour exécuter les commandes du projet.

### Installation avec [`uv`](https://docs.astral.sh/uv/)

Pour installer les dépendances de tous les défis, utilisez la commande suivante à la racine du projet :

```bash
uv sync --all-packages
```

Pour installer les dépendances d'un défi spécifique, utilisez la commande suivante en remplaçant le nom du défi par celui que vous souhaitez :

```bash
uv sync --package <nom_du_défi>
```

## Ressources supplémentaires

### Outils de développement

- [Documentation pytest](https://docs.pytest.org/en/stable/)
- [Documentation pre-commit](https://pre-commit.com/)
- [Documentation Ruff](https://docs.astral.sh/ruff/)
- [Documentation pre-commit Ruff hooks](https://github.com/astral-sh/ruff-pre-commit)
- [PEP 8 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

### Gestion d'environnement

- [uv - Documentation](https://docs.astral.sh/uv/)
