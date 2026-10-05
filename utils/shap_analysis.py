import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
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
    - Logistic Regression inside Pipeline
    - Random Forest inside Pipeline
    - SHAP old API
    - SHAP new API
    """

    # =================================================
    # PIPELINE
    # =================================================

    if isinstance(model, Pipeline):

        # ---------------------------------------------
        # Get final classifier
        # ---------------------------------------------

        final_model = model.named_steps[
            "classifier"
        ]

        # ---------------------------------------------
        # Get preprocessing steps
        # ---------------------------------------------

        preprocessing = model[:-1]

        # Apply preprocessing to X
        X_transformed = preprocessing.transform(X)

        # Convert transformed data back to DataFrame
        # while keeping original gene names
        X_transformed = pd.DataFrame(
            X_transformed,
            columns=X.columns,
            index=X.index
        )

        # =============================================
        # PIPELINE + LOGISTIC REGRESSION
        # =============================================

        if isinstance(
            final_model,
            LogisticRegression
        ):

            explainer = shap.LinearExplainer(
                final_model,
                X_transformed
            )

            try:

                # New SHAP API
                shap_values = explainer(
                    X_transformed
                )

            except Exception:

                # Old SHAP API
                shap_values = explainer.shap_values(
                    X_transformed
                )

            return shap_values, explainer

        # =============================================
        # PIPELINE + RANDOM FOREST
        # =============================================

        elif isinstance(
            final_model,
            RandomForestClassifier
        ):

            explainer = shap.TreeExplainer(
                final_model
            )

            try:

                # New SHAP API
                shap_values = explainer(
                    X_transformed
                )

            except Exception:

                # Old SHAP API
                shap_values = explainer.shap_values(
                    X_transformed
                )

            return shap_values, explainer

        # =============================================
        # UNKNOWN PIPELINE MODEL
        # =============================================

        else:

            raise ValueError(
                "Unsupported pipeline classifier: "
                f"{type(final_model)}"
            )

    # =================================================
    # DIRECT RANDOM FOREST
    # =================================================

    elif isinstance(
        model,
        RandomForestClassifier
    ):

        explainer = shap.TreeExplainer(
            model
        )

        try:

            # New SHAP API
            shap_values = explainer(X)

        except Exception:

            # Old SHAP API
            shap_values = explainer.shap_values(X)

        return shap_values, explainer

    # =================================================
    # DIRECT LOGISTIC REGRESSION
    # =================================================

    elif isinstance(
        model,
        LogisticRegression
    ):

        explainer = shap.LinearExplainer(
            model,
            X
        )

        try:

            # New SHAP API
            shap_values = explainer(X)

        except Exception:

            # Old SHAP API
            shap_values = explainer.shap_values(X)

        return shap_values, explainer

    # =================================================
    # UNSUPPORTED MODEL
    # =================================================

    else:

        raise ValueError(
            f"Unsupported model type: {type(model)}"
        )


# =====================================================
# EXTRACT SHAP VALUES
# =====================================================

def extract_values(shap_values):
    """
    Convert different SHAP output formats into
    a (samples × features) NumPy array.
    """

    # ------------------------------------
    # OLD SHAP API
    # list(class0, class1)
    # ------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) == 2:

            values = shap_values[1]

        else:

            values = shap_values[0]

    # ------------------------------------
    # NEW SHAP Explanation object
    # ------------------------------------

    elif hasattr(
        shap_values,
        "values"
    ):

        values = shap_values.values

    # ------------------------------------
    # Already NumPy array
    # ------------------------------------

    else:

        values = shap_values

    values = np.asarray(
        values
    )

    # ------------------------------------
    # Shape handling
    # ------------------------------------

    # (samples, features, classes)

    if values.ndim == 3:

        # Binary classification
        # Select positive class
        values = values[:, :, 1]

    # ------------------------------------
    # Single sample
    # ------------------------------------

    elif values.ndim == 1:

        values = values.reshape(
            1,
            -1
        )

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
    ).mean(
        axis=0
    )

    importance_df = pd.DataFrame(
        {
            "Gene": X.columns,
            "SHAP Importance": importance
        }
    )

    importance_df = importance_df.sort_values(
        by="SHAP Importance",
        ascending=False
    ).reset_index(
        drop=True
    )

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

    plt.close(
        "all"
    )

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

    plt.close(
        "all"
    )

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
    ).reset_index(
        drop=True
    )
