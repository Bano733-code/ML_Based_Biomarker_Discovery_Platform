import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# =====================================================
# CALCULATE SHAP VALUES
# =====================================================

def calculate_shap_values(model, X):
    """
    Calculate SHAP values for supported models.

    Supports:
    - Random Forest
    - Logistic Regression
    - SHAP old API
    - SHAP new API
    """
   
    if hasattr(model, "steps"):

        final_model = model.steps[-1][1]

        # ---------------------------------------------
        # Pipeline + Logistic Regression
        # ---------------------------------------------

        if isinstance(final_model, LogisticRegression):

            # Transform X using all preprocessing steps
            preprocessing = model[:-1]

            X_transformed = preprocessing.transform(X)

            # Convert transformed data to DataFrame
            # while keeping original gene names
            X_transformed = pd.DataFrame(
                X_transformed,
                columns=X.columns,
                index=X.index
            )

            explainer = shap.LinearExplainer(
                final_model,
                X_transformed
            )

            shap_values = explainer(X_transformed)

            return shap_values, explainer

        # ---------------------------------------------
        # Pipeline + Random Forest
        # ---------------------------------------------

        elif isinstance(final_model, RandomForestClassifier):

            preprocessing = model[:-1]

            X_transformed = preprocessing.transform(X)

            X_transformed = pd.DataFrame(
                X_transformed,
                columns=X.columns,
                index=X.index
            )

            explainer = shap.TreeExplainer(
                final_model
            )

            shap_values = explainer(X_transformed)

            return shap_values, explainer

        else:

            raise ValueError(
                f"Unsupported pipeline final model: "
                f"{type(final_model)}"
            )

    elif isinstance(model, RandomForestClassifier):

        explainer = shap.TreeExplainer(model)

    elif isinstance(model, LogisticRegression):

        explainer = shap.LinearExplainer(
            model,
            X
        )

    else:

        raise ValueError(
            f"Unsupported model type: {type(model)}"
        )

    try:

        # New SHAP API
        shap_values = explainer(X)

    except Exception:

        # Older SHAP API
        shap_values = explainer.shap_values(X)

    return shap_values, explainer


# =====================================================
# EXTRACT SHAP VALUES
# =====================================================

def extract_values(shap_values):
    """
    Convert every SHAP output format into
    a (samples × features) NumPy array.
    """

    # ------------------------------------
    # OLD SHAP
    # list(class0,class1)
    # ------------------------------------

    if isinstance(shap_values, list):

        if len(shap_values) == 2:

            values = shap_values[1]

        else:

            values = shap_values[0]

    # ------------------------------------
    # NEW SHAP Explanation object
    # ------------------------------------

    elif hasattr(shap_values, "values"):

        values = shap_values.values

    # ------------------------------------
    # Already ndarray
    # ------------------------------------

    else:

        values = shap_values

    values = np.asarray(values)

    # ------------------------------------
    # Shape handling
    # ------------------------------------

    # (samples, features, classes)

    if values.ndim == 3:

        values = values[:, :, 1]

    # (features,)

    elif values.ndim == 1:

        values = values.reshape(1, -1)

    return values


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def get_feature_importance(
    X,
    shap_values
):
    """
    Compute mean absolute SHAP value
    for each gene.
    """

    values = extract_values(
        shap_values
    )

    importance = np.abs(
        values
    ).mean(axis=0)

    importance_df = pd.DataFrame(

        {

            "Gene": X.columns,

            "SHAP Importance": importance

        }

    )

    importance_df = importance_df.sort_values(

        by="SHAP Importance",

        ascending=False

    ).reset_index(drop=True)

    return importance_df


# =====================================================
# SHAP SUMMARY PLOT
# =====================================================

def shap_summary_plot(
    shap_values,
    X
):
    """
    Generate SHAP summary plot.
    """

    values = extract_values(
        shap_values
    )

    plt.close("all")

    fig = plt.figure(
        figsize=(10, 6)
    )

    shap.summary_plot(

        values,

        X,

        show=False

    )

    plt.tight_layout()

    return fig


# =====================================================
# SHAP BAR PLOT
# =====================================================

def shap_bar_plot(
    shap_values,
    X
):
    """
    Generate SHAP feature importance bar plot.
    """

    values = extract_values(
        shap_values
    )

    plt.close("all")

    fig = plt.figure(
        figsize=(10, 6)
    )

    shap.summary_plot(

        values,

        X,

        plot_type="bar",

        show=False

    )

    plt.tight_layout()

    return fig


# =====================================================
# LOCAL EXPLANATION
# =====================================================

def explain_single_sample(
    shap_values,
    X,
    sample_index=0
):
    """
    Return SHAP values for one sample.
    Useful for waterfall or force plots.
    """

    values = extract_values(
        shap_values
    )

    return pd.DataFrame(

        {

            "Gene": X.columns,

            "SHAP Value": values[sample_index]

        }

    ).sort_values(

        by="SHAP Value",

        key=np.abs,

        ascending=False

    ).reset_index(drop=True)
