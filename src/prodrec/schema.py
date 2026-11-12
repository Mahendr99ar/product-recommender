"""Column names and meanings for the Santander Product Recommendation data.

The original column names are Spanish abbreviations. Everything downstream uses the
original names so the code matches the Kaggle data; this file is the translation.
"""

from __future__ import annotations

# 24 products, in the order Kaggle lists them. English names follow the Kaggle data page.
PRODUCTS = {
    "ind_ahor_fin_ult1": "Saving account",
    "ind_aval_fin_ult1": "Guarantees",
    "ind_cco_fin_ult1": "Current account",
    "ind_cder_fin_ult1": "Derivatives account",
    "ind_cno_fin_ult1": "Payroll account",
    "ind_ctju_fin_ult1": "Junior account",
    "ind_ctma_fin_ult1": "Mas particular account",
    "ind_ctop_fin_ult1": "Particular account",
    "ind_ctpp_fin_ult1": "Particular plus account",
    "ind_deco_fin_ult1": "Short-term deposit",
    "ind_deme_fin_ult1": "Medium-term deposit",
    "ind_dela_fin_ult1": "Long-term deposit",
    "ind_ecue_fin_ult1": "e-account",
    "ind_fond_fin_ult1": "Funds",
    "ind_hip_fin_ult1": "Mortgage",
    "ind_plan_fin_ult1": "Pension plan",
    "ind_pres_fin_ult1": "Loan",
    "ind_reca_fin_ult1": "Taxes",
    "ind_tjcr_fin_ult1": "Credit card",
    "ind_valo_fin_ult1": "Securities",
    "ind_viv_fin_ult1": "Home account",
    "ind_nomina_ult1": "Payroll",
    "ind_nom_pens_ult1": "Pensions",
    "ind_recibo_ult1": "Direct debit",
}
PRODUCT_COLS = list(PRODUCTS)
SHORT = {c: c.replace("ind_", "").replace("_fin_ult1", "").replace("_ult1", "") for c in PRODUCT_COLS}

DATE = "fecha_dato"        # snapshot month
CUSTOMER = "ncodpers"      # customer id

# Customer attributes used as features (known at the start of the month).
NUMERIC = {
    "age": "Age",
    "antiguedad": "Seniority (months)",
    "renta": "Gross household income",
}
CATEGORICAL = {
    "ind_empleado": "Employee index",
    "pais_residencia": "Country of residence",
    "sexo": "Sex",
    "ind_nuevo": "New customer (joined in last 6 months)",
    "indrel": "Primary customer",
    "indrel_1mes": "Customer type at start of month",
    "tiprel_1mes": "Relation type at start of month",
    "indresi": "Resident",
    "indext": "Foreigner",
    "canal_entrada": "Joining channel",
    "indfall": "Deceased",
    "nomprov": "Province",
    "ind_actividad_cliente": "Active customer",
    "segmento": "Segment",
}
CUSTOMER_COLS = list(NUMERIC) + list(CATEGORICAL)
