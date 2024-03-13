import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

def generate_data(seed, noise_level):
    """
    Generates random data for linear regression with specified noise level.
    """
    np.random.seed(seed)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + noise_level * np.random.randn(100, 1)
    return X, y

def plot_regression_and_stats(X, y, title):
    """
    Plots the linear regression model along with the data points and returns the model for statistics.
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    fig, ax = plt.subplots()
    ax.scatter(X, y, label='Data points')
    ax.plot(X, model.predict(X_with_const), color='red', label='Regression line')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

    return model

def calculate_pvalues(model):
    """
    Manually calculate p-values for model coefficients.
    """
    degrees_of_freedom = len(model.model.endog) - model.df_model - 1
    tvalues = model.tvalues
    pvalues = [2*(1 - stats.t.cdf(np.abs(t), degrees_of_freedom)) for t in tvalues]
    return pvalues

def create_summary_table(model):
    """
    Creates a summary table from a regression model's statistics.
    """
    conf_int = model.conf_int()
    std_err = model.bse
    pvalues = calculate_pvalues(model)

    summary_df = pd.DataFrame({
        "Coefficient": model.params,
        "Standard Error": std_err,
        "t-value": model.tvalues,
        "P-value": pvalues,
        "Lower 95%": conf_int[0],
        "Upper 95%": conf_int[1]
    })

    summary_df.index = ['const (Intercept)', 'x1']

    # Formatting for at least 5 decimal points
    summary_df = summary_df.applymap(lambda x: f"{x:.5f}")

    return summary_df

st.title('Linear Regression Examples with Different Data Densities')

# Generating and plotting data, no noise
X_original, y_original = generate_data(seed=0, noise_level=0)
model_original = plot_regression_and_stats(X_original, y_original, 'Original Regression with Minimal Noise')

# Generating and plotting data with little noise
X_tight, y_tight = generate_data(seed=1, noise_level=0.5)
model_tight = plot_regression_and_stats(X_tight, y_tight, 'Tight Regression')

# Generating and plotting loosely packed data points with increased noise
X_loose, y_loose = generate_data(seed=2, noise_level=10)  # Increased noise level significantly
model_loose = plot_regression_and_stats(X_loose, y_loose, 'Loose Regression with High Noise')

# Displaying regression statistics
st.subheader("Regression Statistics")

summary_df_original = create_summary_table(model_original)
st.write("### Original Regression with Minimal Noise")
st.dataframe(summary_df_original)

summary_df_tight = create_summary_table(model_tight)
st.write("### Tight Regression")
st.dataframe(summary_df_tight)

summary_df_loose = create_summary_table(model_loose)
st.write("### Loose Regression")
st.dataframe(summary_df_loose)

st.caption("""
- **Coefficient**: The effect size of each predictor.
- **Standard Error**: The standard deviation of the coefficient estimate.
- **t-value**: A measure of how statistically significant the coefficient is.
- **P-value**: The probability of observing the data if the null hypothesis (no effect) is true.
- **Lower 95% & Upper 95%**: The confidence interval bounds for the coefficient estimates.
- **const (Intercept)**: The model's intercept. It represents the expected value of the dependent variable when all predictors are zero.
- **x1**: Represents the slope of the regression line, showing the change in the dependent variable for a one-unit change in the predictor.
""")

