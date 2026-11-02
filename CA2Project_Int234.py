# INDIA CRIME DATA ANALYSIS PROJECT

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, log_loss
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    BaggingRegressor, GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 1. DATASET PATH
file_path = r"C:\Users\hp\Downloads\Crime Dataset.csv"
output_folder = r"C:\Users\hp\Downloads\crime_project_outputs"
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Give long column names shorter names
df = df.rename(columns={
    "Crime Head Ipc (Indian Penal Code) Category": "Category",
    "Crime Head Ipc (Indian Penal Code) Sub-Category": "Subcategory",
    "Incidence Of Ipc (Indian Penal Code) Crimes (UOM:Number), Scaling Factor:1": "Incidents",
    "Victims Of Ipc (Indian Penal Code) Crimes (UOM:Number), Scaling Factor:1": "Victims",
    "Ipc (Indian Penal Code) Crime Rate Per Lakh Person (UOM:PerLakh(100000)), Scaling Factor:1": "Crime_Rate"
})

needed = ["State", "Year", "Category", "Subcategory", "Incidents", "Victims", "Crime_Rate"]
missing = [column for column in needed if column not in df.columns]
if missing:
    raise ValueError("These columns are missing: " + ", ".join(missing))

for column in ["Year", "Incidents", "Victims", "Crime_Rate"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.drop_duplicates()
df = df.dropna(subset=["State", "Year", "Category", "Subcategory", "Incidents"])
df["Incidents"] = df["Incidents"].clip(lower=0)
df["Victims"] = df["Victims"].clip(lower=0)
df["Crime_Rate"] = df["Crime_Rate"].fillna(df["Crime_Rate"].median())
df["Year"] = df["Year"].astype(int)

print("\nDATASET INFORMATION")
print("Dataset shape:", df.shape)
print("Total States/UTs:", df["State"].nunique())
print("Years:", df["Year"].min(), "-", df["Year"].max())
print(df.head())

# This function saves and shows one graph at a time.
def show_graph(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

# -------------------------------------------------
# UNIT I: DATA PREPARATION AND EXPLORATORY ANALYSIS
# -------------------------------------------------
print("\nUNIT I: DATA PREPARATION AND EDA")

state_total = df.groupby("State")["Incidents"].sum().sort_values(ascending=False)
year_total = df.groupby("Year")["Incidents"].sum().sort_index()
category_total = df.groupby("Category")["Incidents"].sum().sort_values(ascending=False)

print("\nTop 10 states by recorded incidents:")
print(state_total.head(10))
print("\nCorrelation:")
print(df[["Year", "Incidents", "Victims", "Crime_Rate"]].corr().round(2))

state_total.head(10).sort_values().plot(kind="barh", figsize=(8, 5), title="Top 10 States/UTs by Recorded Incidents")
plt.xlabel("Recorded incidents")
plt.ylabel("State/UT")
show_graph("unit1_top_states.png")

year_total.plot(kind="line", marker="o", figsize=(7, 4), title="Recorded Crime Incidents by Year")
plt.xlabel("Year")
plt.ylabel("Recorded incidents")
plt.grid(True)
show_graph("unit1_crime_by_year.png")

category_total.head(10).sort_values().plot(kind="barh", figsize=(8, 5), title="Top 10 Crime Categories")
plt.xlabel("Recorded incidents")
show_graph("unit1_top_categories.png")

# Prepare features for machine learning
features = df[["Year", "State", "Category", "Subcategory"]]
X_all = pd.get_dummies(features, drop_first=True, dtype=float)
y_all = df["Incidents"].astype(float)

# Use a sample to make it run faster in Python IDLE
if len(df) > 2500:
    sample_rows = df.sample(n=2500, random_state=42).index
    X = X_all.loc[sample_rows]
    y = y_all.loc[sample_rows]
else:
    X = X_all.copy()
    y = y_all.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# UNIT II: REGRESSION
# -------------------------------------------------
print("\nUNIT II: REGRESSION")

regression_results = []

def evaluate_regression(name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    result = {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted)
    }
    regression_results.append(result)
    print("\n" + name)
    print("MAE:", round(result["MAE"], 2))
    print("MSE:", round(result["MSE"], 2))
    print("RMSE:", round(result["RMSE"], 2))
    print("R2:", round(result["R2"], 3))

# Simple linear regression and polynomial regression use Year only
small_x = df.loc[X.index, ["Year"]]
x_train, x_test, small_y_train, small_y_test = train_test_split(
    small_x, y, test_size=0.2, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(x_train, small_y_train)
simple_pred = simple_model.predict(x_test)
evaluate_regression("Simple Linear Regression (OLS)", small_y_test, simple_pred)

poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(x_train, small_y_train)
poly_pred = poly_model.predict(x_test)
evaluate_regression("Polynomial Regression", small_y_test, poly_pred)

multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)
multiple_pred = multiple_model.predict(X_test)
evaluate_regression("Multiple Linear Regression (OLS)", y_test, multiple_pred)

regression_models = {
    "Random Forest": RandomForestRegressor(n_estimators=30, max_depth=15, random_state=42, n_jobs=1),
    "Bagging": BaggingRegressor(n_estimators=5, random_state=42, n_jobs=1),
    "Boosting": GradientBoostingRegressor(n_estimators=40, random_state=42),
    "Neural Network MLP": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(24,), max_iter=500, early_stopping=True, random_state=42)
    )
}

for name, model in regression_models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    evaluate_regression(name, y_test, prediction)

regression_table = pd.DataFrame(regression_results)
regression_table.to_csv(os.path.join(output_folder, "regression_results.csv"), index=False)

plt.figure(figsize=(6, 5))
plt.scatter(y_test, multiple_pred, alpha=0.5)
max_value = max(float(y_test.max()), float(np.max(multiple_pred)))
plt.plot([0, max_value], [0, max_value], "r--")
plt.title("Actual vs Predicted Incidents")
plt.xlabel("Actual incidents")
plt.ylabel("Predicted incidents")
show_graph("unit2_actual_vs_predicted.png")

plt.figure(figsize=(8, 5))
plt.barh(regression_table["Model"], regression_table["R2"])
plt.title("Regression Models: R-squared Comparison")
plt.xlabel("R-squared score")
show_graph("unit2_regression_comparison.png")

# -------------------------------------------------
# UNIT III: CLASSIFICATION
# -------------------------------------------------
print("\nUNIT III: CLASSIFICATION")

# Label 1 means incidents are at or above the dataset median.
labels = (df["Incidents"] >= df["Incidents"].median()).astype(int)
class_labels = labels.loc[X.index]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, class_labels, test_size=0.2, random_state=42, stratify=class_labels
)

classifiers = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=15, random_state=42, n_jobs=1),
    "Neural Network MLP": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(24,), max_iter=500, early_stopping=True, random_state=42)
    )
}

classification_results = []
random_forest_matrix = None

for name, model in classifiers.items():
    model.fit(Xc_train, yc_train)
    prediction = model.predict(Xc_test)

    result = {
        "Model": name,
        "Accuracy": accuracy_score(yc_test, prediction),
        "Precision": precision_score(yc_test, prediction, zero_division=0),
        "Recall": recall_score(yc_test, prediction, zero_division=0),
        "F1": f1_score(yc_test, prediction, zero_division=0)
    }

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(Xc_test)[:, 1]
        result["Log Loss"] = log_loss(yc_test, model.predict_proba(Xc_test), labels=[0, 1])
        result["ROC-AUC"] = roc_auc_score(yc_test, probability)

    classification_results.append(result)
    print("\n", name)
    print("Accuracy:", round(result["Accuracy"], 3))
    print("Precision:", round(result["Precision"], 3))
    print("Recall:", round(result["Recall"], 3))
    print("F1 score:", round(result["F1"], 3))
    if "Log Loss" in result:
        print("Log Loss:", round(result["Log Loss"], 3))
        print("ROC-AUC:", round(result["ROC-AUC"], 3))

    if name == "Random Forest":
        random_forest_matrix = confusion_matrix(yc_test, prediction, labels=[0, 1])

classification_table = pd.DataFrame(classification_results)
classification_table.to_csv(os.path.join(output_folder, "classification_results.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.barh(classification_table["Model"], classification_table["Accuracy"])
plt.xlim(0, 1)
plt.title("Classification Model Accuracy")
plt.xlabel("Accuracy")
show_graph("unit3_classification_accuracy.png")

plt.figure(figsize=(5, 4))
plt.imshow(random_forest_matrix)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.xticks([0, 1], ["Low", "High"])
plt.yticks([0, 1], ["Low", "High"])
for row in range(2):
    for col in range(2):
        plt.text(col, row, str(random_forest_matrix[row, col]), ha="center", va="center")
plt.colorbar()
show_graph("unit3_confusion_matrix.png")

# -------------------------------------------------
# UNIT IV: CLUSTERING AND ASSOCIATION PATTERNS
# -------------------------------------------------
print("\nUNIT IV: CLUSTERING AND PATTERN DETECTION")

state_year = df.groupby(["State", "Year"], as_index=False).agg(
    Total_Incidents=("Incidents", "sum"),
    Average_Crime_Rate=("Crime_Rate", "mean"),
    Total_Victims=("Victims", "sum")
)

cluster_values = state_year[["Total_Incidents", "Average_Crime_Rate", "Total_Victims"]]
scaled_clusters = StandardScaler().fit_transform(cluster_values)

# Elbow method helps compare possible K values
inertia_values = []
for k in range(1, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(scaled_clusters)
    inertia_values.append(model.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 9), inertia_values, marker="o")
plt.title("K-Means Elbow Method")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
show_graph("unit4_elbow_method.png")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
state_year["KMeans_Cluster"] = kmeans.fit_predict(scaled_clusters)

plt.figure(figsize=(7, 5))
plt.scatter(
    state_year["Total_Incidents"],
    state_year["Average_Crime_Rate"],
    c=state_year["KMeans_Cluster"]
)
plt.title("K-Means Clusters")
plt.xlabel("Total recorded incidents")
plt.ylabel("Average crime rate per lakh")
plt.colorbar(label="Cluster")
show_graph("unit4_kmeans_clusters.png")

# Run agglomerative clustering with different linkages
for linkage in ["single", "complete", "average", "ward"]:
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    state_year["Cluster_" + linkage] = model.fit_predict(scaled_clusters)

state_year.to_csv(os.path.join(output_folder, "clustering_results.csv"), index=False)

# Simple association patterns: crime subcategories appearing in the same state-year
baskets = df.groupby(["State", "Year"])["Subcategory"].apply(
    lambda values: set(values.astype(str))
).tolist()

pair_counts = {}
item_counts = {}
for basket in baskets:
    for item in basket:
        item_counts[item] = item_counts.get(item, 0) + 1
    items = sorted(basket)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pair = (items[i], items[j])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

rules = []
for (item_a, item_b), count in pair_counts.items():
    support = count / max(len(baskets), 1)
    if support >= 0.05:
        rules.append({
            "Subcategory A": item_a,
            "Subcategory B": item_b,
            "Support": support,
            "Confidence A to B": count / max(item_counts[item_a], 1),
            "Confidence B to A": count / max(item_counts[item_b], 1)
        })

pd.DataFrame(rules).to_csv(os.path.join(output_folder, "association_patterns.csv"), index=False)
print("Association patterns saved.")

# -------------------------------------------------
# UNIT V: PCA AND NEURAL NETWORKS
# -------------------------------------------------
print("\nUNIT V: PCA AND NEURAL NETWORKS")

pca = PCA(n_components=2)
pca_values = pca.fit_transform(scaled_clusters)
state_year["PC1"] = pca_values[:, 0]
state_year["PC2"] = pca_values[:, 1]
state_year.to_csv(os.path.join(output_folder, "pca_results.csv"), index=False)

print("PCA variance explained:", round(pca.explained_variance_ratio_.sum(), 3))

plt.figure(figsize=(7, 5))
plt.scatter(state_year["PC1"], state_year["PC2"], c=state_year["KMeans_Cluster"])
plt.title("PCA of State-Year Crime Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="K-Means cluster")
show_graph("unit5_pca_scatter.png")

# -------------------------------------------------
# UNIT VI: MODEL PERFORMANCE
# -------------------------------------------------
print("\nUNIT VI: MODEL PERFORMANCE")

forest = regression_models["Random Forest"]
train_score = r2_score(y_train, forest.predict(X_train))
test_score = r2_score(y_test, forest.predict(X_test))

print("Random Forest training R2:", round(train_score, 3))
print("Random Forest testing R2:", round(test_score, 3))

plt.figure(figsize=(6, 4))
plt.bar(["Training", "Testing"], [train_score, test_score])
plt.title("Random Forest: Training vs Testing R2")
plt.ylabel("R2 score")
show_graph("unit6_training_testing.png")

ensemble_table = regression_table[
    regression_table["Model"].isin(["Random Forest", "Bagging", "Boosting", "Neural Network MLP"])
]

plt.figure(figsize=(7, 4))
plt.barh(ensemble_table["Model"], ensemble_table["R2"])
plt.title("Bagging, Boosting, Random Forest and MLP")
plt.xlabel("R2 score")
show_graph("unit6_model_comparison.png")

print("\nPROJECT COMPLETED SUCCESSFULLY")
print("Each graph was shown in a separate window.")
print("All graph images and result CSV files are saved here:")
print(output_folder)
print("Note: recorded crime figures do not represent every crime that actually occurred.")
