import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load the data
truncated_results = pd.read_csv("auc_results_absolute_truncated_all_length_all_model_size.csv")
relative_results = pd.read_csv("auc_results_relative_truncated_all_length_all_model_size.csv")
untruncated_results = pd.read_csv("auc_results_absolute_untruncated_all_length_all_model_size.csv")

# Set a Seaborn style for the plot
sns.set(style="whitegrid")


def draw_by_dataset(csv_results, dataset_name, plot_title):
    """
    Draw graph for a specific dataset from the csv_results.
    """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="length", y="auc", marker="o")

    # Customize the plot
    plt.title(plot_title, fontsize=16)
    plt.xlabel("Length", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()


# Draw for the specific dataset "ArXiv" with an appropriate title
draw_by_dataset(truncated_results, "ArXiv", "Absolute AUC for ArXiv")


def draw_by_dataset_and_model_size(csv_results, dataset_name, plot_title):
    """
    Draw graph for specific dataset and include all model sizes in one graph.
    """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Use seaborn's lineplot with hue to differentiate model sizes.
    sns.lineplot(data=df_filtered, x="length", y="auc", hue="model_size",
                 marker="o", palette="viridis")

    # Customize the plot
    plt.title(plot_title, fontsize=16)
    plt.xlabel("Length", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Model Size', fontsize=12, title_fontsize='13')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()


# Draw for the specific dataset "ArXiv" with an appropriate title and all model sizes in one plot
draw_by_dataset_and_model_size(truncated_results, "Pile-CC", "Absolute AUC for ArXiv by Model Size")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
truncated_results = pd.read_csv("auc_results_absolute_truncated_all_length_all_model_size.csv")
relative_results = pd.read_csv("auc_results_relative_truncated_all_length_all_model_size.csv")
untruncated_results = pd.read_csv("auc_results_absolute_untruncated_all_length_all_model_size.csv")

# Set a Seaborn style for the plot
sns.set(style="whitegrid")


def draw_by_feature(csv_results, dataset_name, plot_title):
    """
    Draw graph grouped by feature and include all model sizes in one graph.
    """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]

    # Extract unique features
    unique_features = df_filtered['feature'].unique()

    for feature in unique_features:
        df_feature = df_filtered[df_filtered['feature'] == feature]

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Use seaborn's lineplot with hue to differentiate model sizes.
        sns.lineplot(data=df_feature, x="length", y="auc", hue="model_size",
                     marker="o", palette="viridis")

        # Customize the plot
        plt.title(f"{plot_title} - {feature}", fontsize=16)
        plt.xlabel("Length", fontsize=14)
        plt.ylabel("AUC", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Model Size', fontsize=12, title_fontsize='13')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.show()


# Draw for the specific dataset "ArXiv" with an appropriate title and by feature
draw_by_feature(truncated_results, "Github", "Absolute AUC for ArXiv by Model Size and Feature")

sns.set(style="whitegrid")


# Function to plot PDF for a specific category
def plot_pdf_by_category(df, category, title):
    plt.figure(figsize=(12, 8))

    # Use seaborn to plot the density function
    sns.kdeplot(data=df, x="auc", hue=category, shade=True, linewidth=2)

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel("AUC", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title=category, fontsize=12, title_fontsize='13')

    plt.show()


# Plot the PDF for all datasets in one figure
plot_pdf_by_category(untruncated_results, "dataset", "Probability Density Function of AUC for All Datasets")

# Plot the PDF for all model sizes in one figure
plot_pdf_by_category(untruncated_results, "model_size", "Probability Density Function of AUC for All Model Sizes")

# Plot the PDF for all features in one figure
plot_pdf_by_category(untruncated_results, "feature", "Probability Density Function of AUC for All Features")

# Plot the PDF for all lengths in one figure
plot_pdf_by_category(untruncated_results, "length", "Probability Density Function of AUC for All Length")
