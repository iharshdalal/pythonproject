import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.tsa.ardl import ardl_select_order
import statsmodels.api as sm
from arch.univariate import EGARCH, ConstantMean
from arch.univariate import Normal
import warnings
warnings.filterwarnings("ignore")

# File paths
paths = {
    "stock_oil": r"C:\Users\Harsh\Downloads\1-s2.0-S014098831930283X-mmc1\Data and Readme File\Readme File\1-Stock and Oil Data.xlsx",
    "dcc": r"C:\Users\Harsh\Downloads\1-s2.0-S014098831930283X-mmc1\Data and Readme File\Readme File\2-DCC Series.xlsx",
    "adcc": r"C:\Users\Harsh\Downloads\1-s2.0-S014098831930283X-mmc1\Data and Readme File\Readme File\3-ADCC Series.xlsx",
    "dcc_det": r"C:\Users\Harsh\Downloads\1-s2.0-S014098831930283X-mmc1\Data and Readme File\Readme File\4-Data for the analysis of the determinants of the DCCs.xlsx",
    "adcc_det": r"C:\Users\Harsh\Downloads\1-s2.0-S014098831930283X-mmc1\Data and Readme File\Readme File\5-Data for the analysis of the determinants of the ADCCs.xlsx"
}

# Load datasets
dcc_df = pd.read_excel(paths['dcc'])
adcc_df = pd.read_excel(paths['adcc'])
dcc_det_df = pd.read_excel(paths['dcc_det'])
adcc_det_df = pd.read_excel(paths['adcc_det'])

#  Convert date columns
for df in [dcc_df, adcc_df]:
    df['Date'] = pd.to_datetime(df['Date'])

# Columns of interest
dcc_cols = ['eco-pse', 'eco-oilf', 'pse-oilf']


#  Load Dataset 1
# Drop missing and calculate log returns
df = pd.read_excel(paths['stock_oil'])[['ECO', 'PSE', 'Oilf']].dropna()
df['DLECO'] = np.log(df['ECO']).diff()
df['DLPSE'] = np.log(df['PSE']).diff()
df['DLOILF'] = np.log(df['Oilf']).diff()
returns = df[['DLECO', 'DLPSE', 'DLOILF']].dropna()

# === Table 1: Summary Statistics ===
def get_summary(series):
    jb_stat, jb_p = jarque_bera(series)
    return {
        'Mean': series.mean(),
        'Median': series.median(),
        'Max': series.max(),
        'Min': series.min(),
        'Std. Dev.': series.std(),
        'Skewness': skew(series),
        'Kurtosis': kurtosis(series, fisher=False),
        'Jarque-Bera': jb_stat,
        'p-Value': jb_p
    }

summary = {col: get_summary(returns[col]) for col in returns.columns}
summary_df = pd.DataFrame(summary).T.round(6)

print("\n=== Table 1: Summary Statistics ===")
print(summary_df)


# === Table 4 ===
# multivariate module is now not supported in python latest version. Hence  the values are not matching to the paper
# === EGARCH Estimation Function ===
def estimate_egarch(series, asymmetric=True):
    am = ConstantMean(series)
    model = EGARCH(p=1, o=1 if asymmetric else 0, q=1)
    am.volatility = model
    res = am.fit(disp='off')
    params = res.params
    pvalues = res.pvalues
    llf = res.loglikelihood
    return {
        'alpha2': params['alpha[1]'],
        'p_alpha2': pvalues['alpha[1]'],
        'beta2': params['beta[1]'],
        'p_beta2': pvalues['beta[1]'],
        'gamma2': params['gamma[1]'] if asymmetric else np.nan,
        'p_gamma2': pvalues['gamma[1]'] if asymmetric else np.nan,
        'loglik': llf
    }

# Estimate models
adcc_results = [estimate_egarch(returns[col], asymmetric=True) for col in returns.columns]
dcc_results = [estimate_egarch(returns[col], asymmetric=False) for col in returns.columns]

# Average results for formatting
def avg_and_format(results, model_name):
    avg = pd.DataFrame(results).mean()
    row1 = {
        'Model': f'{model_name}-EGARCH(1,1)',
        'α²': round(avg['alpha2'], 6),
        'β²': round(avg['beta2'], 6),
        'γ²': round(avg['gamma2'], 6) if model_name == 'ADCC' else '',
        'Log-likelihood': round(sum([r['loglik'] for r in results]), 2)
    }
    row2 = {
        'Model': 'P value',
        'α²': round(np.mean([r['p_alpha2'] for r in results]), 3),
        'β²': round(np.mean([r['p_beta2'] for r in results]), 3),
        'γ²': round(np.mean([r['p_gamma2'] for r in results]), 3) if model_name == 'ADCC' else '',
        'Log-likelihood': ''
    }
    return [row1, row2]

# Prepare final table
final_table = pd.DataFrame(
    avg_and_format(adcc_results, 'ADCC') + avg_and_format(dcc_results, 'DCC')
)

print("\n=== Table 4: Estimates of DCC and ADCC Models ===\n")
print(final_table.to_string(index=False))



# Summary statistics function for Table 5
def full_summary(df, cols):
    stats = []
    for col in cols:
        data = df[col].dropna()
        jb_stat, jb_p = jarque_bera(data)
        row = {
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Max': np.max(data),
            'Min': np.min(data),
            'Std Dev': np.std(data),
            'Skewness': skew(data),
            'Kurtosis': kurtosis(data) + 3,
            'Jarque-Bera': jb_stat,
            'P-value': jb_p
        }
        stats.append(row)
    return pd.DataFrame(stats, index=[col.upper() for col in cols])

# Display full stats
dcc_stats = full_summary(dcc_df, dcc_cols)
adcc_stats = full_summary(adcc_df, dcc_cols)

print("\n=== DCC SUMMARY STATISTICS ===")
print(dcc_stats.round(4))
print("\n=== ADCC SUMMARY STATISTICS ===")
print(adcc_stats.round(4))

# Plot DCC and ADCC series (Figure 2 & Figure 3)
def plot_series(df, title_prefix):
    for col in dcc_cols:
        plt.figure(figsize=(10, 4))
        plt.plot(df['Date'], df[col], label=col.upper())
        plt.title(f"{title_prefix} - {col.upper()}")
        plt.xlabel("Date")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

plot_series(dcc_df, "DCC")
plot_series(adcc_df, "ADCC")