# Evaluation (real)

Evaluation month: 2016-05-28. 29,717 of 931,453 customers added at least one product (3.19%).

## Models (customers who added something)
| model              |   customers |   map@7 |   precision@1 |   recall@7 |   kaggle_style_map@7 |
|:-------------------|------------:|--------:|--------------:|-----------:|---------------------:|
| popularity         |       29717 |  0.6338 |        0.4172 |     0.9607 |               0.0202 |
| segment_popularity |       29717 |  0.6517 |        0.4557 |     0.9621 |               0.0208 |
| lightgbm           |       29717 |  0.8894 |        0.8237 |     0.9901 |               0.0284 |

`kaggle_style_map@7` averages over all customers, counting those who added nothing as 0, like the competition leaderboard.

## Is the difference real? (paired bootstrap, 95 % interval of the MAP@7 difference)
| comparison                    |   diff |   ci_low |   ci_high |
|:------------------------------|-------:|---------:|----------:|
| lightgbm - popularity         | 0.2557 |   0.2521 |    0.2594 |
| lightgbm - segment_popularity | 0.2377 |   0.234  |    0.2413 |

## Re-activations vs first-time products
A re-activation is a product the customer owned at some point 2 to 6 months before. MRR = mean of 1 / rank of the added product.

|                                         |   count |   top1 |   mrr |   recall@7 |
|:----------------------------------------|--------:|-------:|------:|-----------:|
| ('popularity', 're-activation')         |   19555 |  0.405 | 0.621 |      0.987 |
| ('popularity', 'first time')            |   18342 |  0.245 | 0.496 |      0.941 |
| ('segment_popularity', 're-activation') |   19555 |  0.446 | 0.649 |      0.989 |
| ('segment_popularity', 'first time')    |   18342 |  0.263 | 0.502 |      0.942 |
| ('lightgbm', 're-activation')           |   19555 |  0.756 | 0.87  |      1     |
| ('lightgbm', 'first time')              |   18342 |  0.528 | 0.69  |      0.981 |

## New vs established customers
|                                            |   customers |   map@7 |
|:-------------------------------------------|------------:|--------:|
| ('popularity', 'new (< 6 months)')         |        6039 |  0.4882 |
| ('popularity', 'established')              |       23678 |  0.6709 |
| ('segment_popularity', 'new (< 6 months)') |        6039 |  0.4887 |
| ('segment_popularity', 'established')      |       23678 |  0.6933 |
| ('lightgbm', 'new (< 6 months)')           |        6039 |  0.9014 |
| ('lightgbm', 'established')                |       23678 |  0.8864 |

## LightGBM recall by product
| product           | name                    |   additions |   in_top7 |   recall@7 |
|:------------------|:------------------------|------------:|----------:|-----------:|
| ind_recibo_ult1   | Direct debit            |       10220 |     10219 |      1     |
| ind_nom_pens_ult1 | Pensions                |        5541 |      5527 |      0.997 |
| ind_nomina_ult1   | Payroll                 |        5516 |      5499 |      0.997 |
| ind_cco_fin_ult1  | Current account         |        5511 |      5511 |      1     |
| ind_tjcr_fin_ult1 | Credit card             |        4252 |      4216 |      0.992 |
| ind_ecue_fin_ult1 | e-account               |        2723 |      2707 |      0.994 |
| ind_cno_fin_ult1  | Payroll account         |        2369 |      2361 |      0.997 |
| ind_ctma_fin_ult1 | Mas particular account  |         713 |       696 |      0.976 |
| ind_reca_fin_ult1 | Taxes                   |         283 |       253 |      0.894 |
| ind_ctop_fin_ult1 | Particular account      |         226 |       183 |      0.81  |
| ind_valo_fin_ult1 | Securities              |         183 |       117 |      0.639 |
| ind_ctpp_fin_ult1 | Particular plus account |         131 |        96 |      0.733 |
| ind_ctju_fin_ult1 | Junior account          |          77 |        77 |      1     |
| ind_fond_fin_ult1 | Funds                   |          61 |        20 |      0.328 |
| ind_dela_fin_ult1 | Long-term deposit       |          46 |        44 |      0.957 |
| ind_plan_fin_ult1 | Pension plan            |          22 |         8 |      0.364 |
| ind_pres_fin_ult1 | Loan                    |           7 |         2 |      0.286 |
| ind_viv_fin_ult1  | Home account            |           7 |         1 |      0.143 |
| ind_cder_fin_ult1 | Derivatives account     |           5 |         3 |      0.6   |
| ind_hip_fin_ult1  | Mortgage                |           3 |         0 |      0     |
| ind_ahor_fin_ult1 | Saving account          |           1 |         0 |      0     |

## Top 15 features (gain share)
| feature        |   gain |   share |
|:---------------|-------:|--------:|
| nom_pens_drops | 443319 |  0.0814 |
| recibo_lag1    | 419675 |  0.0771 |
| cco_lag1       | 406180 |  0.0746 |
| n_owned_lag1   | 332611 |  0.0611 |
| tjcr_drops     | 302950 |  0.0556 |
| age            | 256799 |  0.0472 |
| antiguedad     | 248858 |  0.0457 |
| nom_pens_lag1  | 245597 |  0.0451 |
| nom_pens_lag2  | 223822 |  0.0411 |
| recibo_drops   | 223300 |  0.041  |
| nomina_drops   | 212983 |  0.0391 |
| n_drops        | 181625 |  0.0333 |
| cno_lag1       | 130064 |  0.0239 |
| tjcr_lag1      | 124869 |  0.0229 |
| renta          | 104175 |  0.0191 |
