import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, features):
    importance = model.feature_importance_
    plt.figure(figsize=(8,5))
    sns.barplot(x=importance, y=features)
    plt.title("feature Importance")
    plt.show()