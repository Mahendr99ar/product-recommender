# Evaluation (demo)

Synthetic data. These numbers check that the code works. They are not findings.

Evaluation month: 2016-05-28. 1,031 of 20,000 customers added at least one product (5.16%).

## Models (customers who added something)
| model              |   customers |   map@7 |   precision@1 |   recall@7 |   kaggle_style_map@7 |
|:-------------------|------------:|--------:|--------------:|-----------:|---------------------:|
| popularity         |        1031 |  0.6072 |        0.4772 |     0.8938 |               0.0313 |
| segment_popularity |        1031 |  0.6136 |        0.4772 |     0.8986 |               0.0316 |
| lightgbm           |        1031 |  0.6425 |        0.5344 |     0.8826 |               0.0331 |

`kaggle_style_map@7` averages over all customers, counting those who added nothing as 0, like the competition leaderboard.

## Is the difference real? (paired bootstrap, 95 % interval of the MAP@7 difference)
| comparison                    |   diff |   ci_low |   ci_high |
|:------------------------------|-------:|---------:|----------:|
| lightgbm - popularity         | 0.0353 |   0.0211 |    0.0493 |
| lightgbm - segment_popularity | 0.0289 |   0.0147 |    0.0432 |

## New vs established customers
|                                            |   customers |   map@7 |
|:-------------------------------------------|------------:|--------:|
| ('popularity', 'new (< 6 months)')         |         302 |  0.9117 |
| ('popularity', 'established')              |         729 |  0.481  |
| ('segment_popularity', 'new (< 6 months)') |         302 |  0.9068 |
| ('segment_popularity', 'established')      |         729 |  0.4921 |
| ('lightgbm', 'new (< 6 months)')           |         302 |  0.911  |
| ('lightgbm', 'established')                |         729 |  0.5313 |

## LightGBM recall by product
| product           | name                    |   additions |   in_top7 |   recall@7 |
|:------------------|:------------------------|------------:|----------:|-----------:|
| ind_cco_fin_ult1  | Current account         |         271 |       271 |      1     |
| ind_recibo_ult1   | Direct debit            |         239 |       239 |      1     |
| ind_nom_pens_ult1 | Pensions                |          94 |        87 |      0.926 |
| ind_cno_fin_ult1  | Payroll account         |          88 |        86 |      0.977 |
| ind_tjcr_fin_ult1 | Credit card             |          77 |        77 |      1     |
| ind_nomina_ult1   | Payroll                 |          68 |        59 |      0.868 |
| ind_ecue_fin_ult1 | e-account               |          48 |        37 |      0.771 |
| ind_valo_fin_ult1 | Securities              |          41 |        25 |      0.61  |
| ind_dela_fin_ult1 | Long-term deposit       |          37 |        25 |      0.676 |
| ind_fond_fin_ult1 | Funds                   |          24 |        13 |      0.542 |
| ind_ctma_fin_ult1 | Mas particular account  |          19 |         3 |      0.158 |
| ind_ctop_fin_ult1 | Particular account      |          11 |         0 |      0     |
| ind_plan_fin_ult1 | Pension plan            |           9 |         1 |      0.111 |
| ind_reca_fin_ult1 | Taxes                   |           7 |         2 |      0.286 |
| ind_hip_fin_ult1  | Mortgage                |           5 |         0 |      0     |
| ind_deme_fin_ult1 | Medium-term deposit     |           3 |         0 |      0     |
| ind_ctpp_fin_ult1 | Particular plus account |           3 |         0 |      0     |
| ind_pres_fin_ult1 | Loan                    |           2 |         0 |      0     |
| ind_deco_fin_ult1 | Short-term deposit      |           2 |         0 |      0     |
| ind_ctju_fin_ult1 | Junior account          |           1 |         0 |      0     |

## Top 15 features (gain share)
| feature       |     gain |   share |
|:--------------|---------:|--------:|
| antiguedad    | 37194.9  |  0.2711 |
| renta         | 16877.6  |  0.123  |
| age           | 14186.2  |  0.1034 |
| cno_lag1      |  7919.51 |  0.0577 |
| nomprov       |  6032.64 |  0.044  |
| ahor_lag1     |  5846.13 |  0.0426 |
| recibo_lag1   |  5346.91 |  0.039  |
| canal_entrada |  5302.92 |  0.0387 |
| recibo_lag2   |  2797.51 |  0.0204 |
| n_owned_lag1  |  2782.95 |  0.0203 |
| cco_lag1      |  2764.99 |  0.0202 |
| aval_lag1     |  2378.72 |  0.0173 |
| nomina_lag1   |  2341.71 |  0.0171 |
| nom_pens_lag1 |  2152.63 |  0.0157 |
| cno_lag2      |  1913.09 |  0.0139 |
