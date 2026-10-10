# Social Commerce Usage Analytics

An end-to-end data science pipeline that segments Gen Z social commerce users and predicts how much they actually use social commerce platforms, built with tested Python modules and a GitHub Actions CI/CD pipeline.

![CI](https://github.com/ulannn14/social-commerce-usage-analytics/actions/workflows/ci-cd.yml/badge.svg)

## Results at a glance

- **Prediction:** A tuned Random Forest explains 75% of the variance in actual usage behavior on a held-out test set (R² = 0.752, RMSE = 0.361 on a 5-point scale), ahead of a tuned MLP (R² = 0.710).
- **Segmentation:** K-Means finds four user segments whose usage differs significantly (Kruskal-Wallis H = 421.32, p < .001; all pairs significant in Dunn's post hoc test).
- **Key drivers:** Perceived ease of use and perceived usefulness account for about half of the Random Forest's feature importance.
- **Engineering:** 66 automated tests (unit, integration, and system) run on every push and pull request.

## The question

Which perceptions drive actual social commerce usage among Gen Z university students, and can users be grouped into meaningful segments?

The data is a published survey of 757 Gen Z university students in Vietnam (ages 18 to 27) with 31 variables: four demographics and Likert-scale items measuring seven constructs.

| Code | Construct | Items |
| --- | --- | --- |
| PU | Perceived Usefulness | 4 |
| PEU | Perceived Ease of Use | 3 |
| FSC | Familiarity with Social Commerce | 3 |
| SP | Social Presence | 4 |
| TP | Trust in Platform | 3 |
| IB | Interaction Behavior | 4 |
| AUB | Actual Usage Behavior (target) | 4 |

## Pipeline

1. **Data validation and cleaning.** Checks schema, data types, missing values, duplicates, and value ranges. Drops two undocumented or zero-variance columns (`PEU4`, `Job`).
2. **Feature engineering.** Averages each construct's items into one composite score per respondent.
3. **Exploratory analysis.** Descriptive statistics, Pearson correlations, Welch's t-test (usage by gender), and one-way ANOVA (usage by area of residence).
4. **Clustering.** Standardizes the six predictor constructs, checks multicollinearity with VIF, and fits K-Means.
5. **Statistical inference.** Shapiro-Wilk for normality, then Kruskal-Wallis and Dunn's post hoc test (Bonferroni) to compare usage across clusters.
6. **Machine learning.** 80/20 train-test split, scaling fitted on training data only, and GridSearchCV with 10-fold cross-validation for Random Forest and MLP regressors.

## Findings

### User segments

| Segment | Respondents | Mean usage (AUB) |
| --- | --- | --- |
| Highly engaged users | 97 | 4.47 |
| Regular users | 382 | 3.92 |
| Moderate or occasional users | 258 | 3.17 |
| Low engagement users | 20 | 1.71 |

Four clusters were chosen from the elbow in inertia and because the four segments are distinct and interpretable. The silhouette score is highest at k = 2 (0.436, versus 0.377 at k = 4), so k = 4 trades some cluster separation for more useful segments.

### Model comparison (held-out test set, n = 152)

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Random Forest | 0.230 | 0.361 | 0.752 |
| MLP | 0.263 | 0.391 | 0.710 |

Best Random Forest: 300 trees, 2 features per split, max depth 10, min samples split 5, min samples leaf 2.
Best MLP: one hidden layer of 100 neurons, tanh activation, learning rate 0.001, alpha 0.001.

### Feature importance (Random Forest)

| Construct | Importance |
| --- | --- |
| Perceived Ease of Use | 0.27 |
| Perceived Usefulness | 0.25 |
| Familiarity with Social Commerce | 0.19 |
| Social Presence | 0.11 |
| Interaction Behavior | 0.09 |
| Trust in Platform | 0.08 |

Feature importance shows what the model relied on, not causation.

### Demographics

Usage did not differ significantly by gender (Welch's t = -0.42, p = 0.68) or by area of residence (ANOVA F = 1.47, p = 0.23).

## Repository structure

```
.github/workflows/ci-cd.yml   CI/CD pipeline
ml/
  datasets/                   Survey dataset (CSV)
  notebooks/                  Analysis notebooks (EDA, clustering and statistics, machine learning)
  scripts/
    preprocessing.py          Loading, validation, cleaning, composite scores, EDA statistics
    clustering.py             Scaling, VIF, K-Means, cluster profiles, non-parametric tests
    machinelearning.py        Splitting, scaling, grid search, prediction, evaluation
  tests/
    test_preprocessing.py     28 unit tests
    test_clustering.py        10 unit tests
    test_machinelearning.py   24 unit tests
    test_integration.py       4 integration and system tests
requirements.txt
```

## Getting started

Requires Python 3.11.

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt
```

Run the full test suite from the repository root:

```bash
PYTHONPATH=. python -m pytest ml/tests/ -v
```

The machine learning and integration tests run full grid searches, so they can take several minutes.

To reproduce the analysis, open the notebooks in `ml/notebooks/` in order.

## Testing and CI/CD

The notebook logic lives in importable modules under `ml/scripts/`, so every step can be tested on its own.

- **Unit tests** check each function against known values from the real dataset, for example exact respondent counts, composite scores for specific rows, correlation coefficients, and test statistics.
- **Integration tests** run each stage end to end (preprocessing and EDA, clustering, machine learning).
- **A system test** runs the whole pipeline from raw CSV to model comparison in one flow.
- **Regression checks** guard against a past bug where the Social Presence score was computed from three items instead of four.

The GitHub Actions workflow runs on every push and pull request to `development` and `main`. It installs dependencies, runs the unit tests, runs the integration and system tests, then packages the scripts, tests, dataset, and notebooks into a versioned release.

## Limitations

- The sample covers one age group in one country, and 96% of respondents use social media daily, so results may not generalize.
- Respondents were recruited voluntarily, which can over-represent students already active in social commerce.
- All predictors are self-reported perceptions and are strongly correlated with each other (r = 0.64 to 0.80).
- 129 rows are exact duplicates. They were kept because the dataset has no respondent ID and identical Likert answers are plausible.

## Dataset

Wang, C.-K., Lin, C.-Y., & Le, B. H. (2026). Dataset on social and behavioral factors influencing social commerce adoption among Gen Z university students in Vietnam. *Data in Brief, 65*, Article 112529. https://doi.org/10.1016/j.dib.2026.112529

## Team
**Rlsrain Mackenlhy Orcullo** led the project, chose the modeling approach, planned the machine learning pipeline and the CI/CD workflow, performed the statistical analysis, and wrote the preprocessing and EDA test suite.
**Lian Carlos Barte**
**Leigh Gwyneth Albo**
