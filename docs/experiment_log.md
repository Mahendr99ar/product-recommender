# Experiment log

One entry per experiment, including the ones that did not work.

## Template

```
### YYYY-MM-DD: short title
Question:   what I wanted to find out
Change:     the one thing I changed
Data:       demo or Kaggle, training months, evaluation month
Result:     MAP@7 with the baseline and the bootstrap interval next to it
Conclusion: keep / revert / unclear, and why
Next:       what this suggests trying
```

## Entries

### Synthetic data: new vs established customers
Question:   Where does LightGBM gain over popularity?
Change:     split the evaluation by seniority (< 6 months vs the rest)
Data:       demo, evaluation month 2016-05-28
Result:     new customers 0.912 (popularity) vs 0.911 (LightGBM); established 0.481 vs 0.531
Conclusion: the gain is all in established customers, whose past products carry information
Next:       check whether the same holds on the Kaggle data

### Synthetic data: which training months
Question:   Last month, same month last year, or several months?
Change:     training months only (configs/config.yaml, `experiments`)
Data:       demo, evaluation month 2016-05-28
Result:     all five choices between 0.638 and 0.653 MAP@7; every gain over popularity has a
            95 % bootstrap interval above zero; differences between choices are smaller than
            the interval widths (about 0.03)
Conclusion: unclear on synthetic data; the demo has no May seasonality to find
Next:       repeat on the Kaggle data, and add an experiment that predicts June 2015

### Kaggle data: baseline run
Question:   How much better than popularity is LightGBM on real customers?
Change:     none, default config
Data:       Kaggle, 6 training months, evaluation month 2016-05-28 (29,717 customers added)
Result:     MAP@7 0.889 vs 0.634 (popularity); difference +0.256, 95 % interval [+0.252, +0.259]
Conclusion: keep; Kaggle-style 0.0284 against a ceiling of 0.0319
Next:       the top feature was nom_pens_drops; check how many additions are re-activations

### Kaggle data: re-activations
Question:   Is the model mostly predicting products that come back?
Change:     split additions by whether the product was owned 2 to 6 months earlier
Data:       Kaggle, evaluation month 2016-05-28
Result:     51.6 % of additions are re-activations; first-time products: top-1 52.8 % (LightGBM)
            vs 24.5 % (popularity)
Conclusion: re-activations inflate the headline MAP, but the model is also much better on
            first-time products
Next:       a separate model or ranking for first-time products only

### Kaggle data: training months
Question:   Last month, same month last year, or several months?
Change:     training months only
Data:       Kaggle, evaluation month 2016-05-28
Result:     last month only 0.895 (16 s); last year + last month 0.897 (30 s); six months 0.889
            (183 s); May 2015 only 0.880
Conclusion: one recent month is enough; differences of about 0.01 not yet tested between rows
Next:       paired bootstrap between experiments; predict June 2015 to test the June effect
