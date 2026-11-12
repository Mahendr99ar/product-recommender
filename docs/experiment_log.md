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
