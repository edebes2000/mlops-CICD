# src/infer.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Separate inference so it can be reused for batch or real time prediction
- Responsibility (separation of concerns): Only runs model.predict and formats outputs, no training or metrics
- Pipeline contract (inputs and outputs): Inputs are a model and inference features, output is a predictions DataFrame

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: Fitted model with predict method
    - X_infer: Feature DataFrame for inference
    Outputs:
    - df_pred: DataFrame with one column named prediction
    Why this contract matters for reliable ML delivery:
    - A consistent prediction schema reduces downstream integration risk
    """
    print("[infer.run_inference] Running inference")  # TODO: replace with logging later

    preds = model.predict(X_infer)
    df_pred = pd.DataFrame({"prediction": preds})

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace or extend the baseline logic here
    # Why: Postprocessing depends on business thresholds and how predictions are consumed
    # Examples:
    # 1. Convert probabilities to labels at a business threshold
    # 2. Clip regression outputs to valid ranges
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return df_pred