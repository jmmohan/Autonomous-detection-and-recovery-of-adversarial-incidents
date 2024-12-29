#### visualization.py

import shap
import matplotlib.pyplot as plt

def visualize_explanations(shap_values, data):
    shap.summary_plot(shap_values, data)
