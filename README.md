# Disaster Tweets: NLP Classification

Notebook: [disaster_tweets_nlp.ipynb](disaster_tweets_nlp.ipynb) (text in Portuguese)
Data: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) on Kaggle, 7,613 labelled tweets.

The task is to tell tweets about real disasters from tweets that use disaster vocabulary figuratively ("this party is on fire"). The classes are mildly imbalanced, so I used F1 as the main metric.

## What I did

**Metadata columns.** `keyword` has 61 nulls; `location` has 33% nulls, 4,521 distinct values and inconsistent spelling (`USA`, `United States`, `U.S.A`, plus emojis, coordinates and jokes). I normalised location into 21 geographic groups, which turned out to carry signal: groups like Pakistan, Nigeria and India have a higher share of real-disaster tweets. For keyword I imputed by regex: the 221 known keywords sorted longest first, single words matched on word boundaries, multi-word phrases as exact substrings. The 16 tweets with no match at all were all non-disaster, so "no keyword found" became a feature in itself.

**Text features.** Real-disaster tweets are slightly longer (about 103 vs 96 characters), more often contain a URL (people share news sources), and use more hashtags. I built `char_len`, `has_url`, `hashtag_count`, `capital_ratio`, `keyword_was_null` and `keyword_in_text`, and stacked them as sparse columns next to a TF-IDF matrix of the cleaned text (HTML unescaped; URLs, handles, numbers and punctuation removed; lowercased; stopwords dropped; lemmatised with WordNet).

**Models.** Logistic Regression, Random Forest, LightGBM, and LightGBM tuned with Optuna (25 trials). The tuned LightGBM was then checked with 10-fold CV.

## Results

Hold-out set (1,523 tweets):

| Model | F1 | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.759 | 0.813 | 0.711 | 0.866 |
| LightGBM (Optuna) | 0.747 | 0.804 | 0.697 | 0.849 |
| Random Forest | 0.725 | 0.864 | 0.624 | 0.858 |
| LightGBM (default) | 0.714 | 0.745 | 0.687 | 0.831 |

Tuned LightGBM, 10-fold CV: F1 0.729 ± 0.016. Kaggle public score with the tuned LightGBM: 0.782.

Logistic regression is marginally ahead on this hold-out, which is common for sparse TF-IDF inputs; I submitted the tuned LightGBM because it was the model I had validated with 10-fold CV. The coefficients tell a consistent story: `wildfire`, `flood`, `earthquake`, `casualti`, `debris` and `has_url` push towards disaster, while emotional hyperbole and everyday words push away.

**Error analysis.** False negatives are short, link-free, colloquial reports of real events ("omg earthquake"). False positives are figurative tweets with the formal structure of an emergency tweet (links, hashtags, length).

## What I would improve

The city-to-country mapping for `location` was done by hand; a gazetteer such as `geonamescache` would scale it. The error analysis points at meaning rather than surface features, so the next step is a pretrained transformer (DistilBERT or similar), which tends to score around 0.83–0.84 on this competition.

---

This case is one of seven in my [data science portfolio](https://github.com/juliapmonteirojm-lab/data-science-portfolio).
