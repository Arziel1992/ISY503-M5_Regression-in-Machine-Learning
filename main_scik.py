import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression


def generate_data(seed):
    """
    Generates random data for linear regression.

    Parameters:
    - seed: An integer to seed the random number generator for reproducibility.

    Returns:
    - X: Features for linear regression.
    - y: Target values for linear regression.
    """
    np.random.seed(seed)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y

def plot_regression(X, y, model):
    """
    Plots the linear regression model along with the data points.

    Parameters:
    - X: Features for linear regression.
    - y: Target values for linear regression.
    - model: The trained linear regression model.
    """
    fig, ax = plt.subplots()
    ax.scatter(X, y, label='Data points')
    ax.plot(X, model.predict(X), color='red', label='Regression line')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Simple Linear Regression Visualization')
    ax.legend()
    return fig

st.title('Simple Linear Regression Example')

# User interface for generating new data
seed = np.random.randint(0, 10000) if st.button('Generate New Data') else 0

# Data generation and model training
X, y = generate_data(seed)
model = LinearRegression()
model.fit(X, y)

# Plotting
fig = plot_regression(X, y, model)
st.pyplot(fig)
st.write("Visualization includes randomly generated data points and the fitted linear regression line.")

# Displaying model parameters
intercept = model.intercept_[0]
coefficient = model.coef_[0][0]
st.write(f"Intercept: {intercept:.2f}", "The y-intercept of the regression line.")
st.write(f"Coefficient: {coefficient:.2f}", "The slope of the regression line, representing the relationship between X and y.")

# Explanation of the linear regression equation
st.write(f"The linear regression equation is modeled as `y = {intercept:.2f} + {coefficient:.2f}X`.")
