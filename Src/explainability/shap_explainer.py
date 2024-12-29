### 3. Explainability and Transparency Module (ETM)
#### shap_explainer.py

import shap

def generate_shap_explanations(model, data):
    explainer = shap.Explainer(model, data)
    shap_values = explainer(data)
    return shap_values
