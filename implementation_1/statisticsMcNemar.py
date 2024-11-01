import sys
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar
import seaborn as sns
import numpy as np
import itertools
from read_data import X_train, y_train, X_test, y_test
from read_data import X_train_bigram, X_test_bigram

n_tests = sys.argv[1] if len(sys.argv) >= 2 else 200
graph_res = sys.argv[2] if len(sys.argv) >= 3 else True


X_train_unigram, X_test_unigram = X_train, X_test


models = {
    "NB_Uni": MultinomialNB(alpha= 0.25),
    "NB_Bi": MultinomialNB(alpha= 0.1),
    "LogReg_Uni": LogisticRegression(C= 25, penalty='l1', solver='liblinear'),
    "LogReg_Bi": LogisticRegression(C= 100, penalty='l1', solver='liblinear'),
    "DecTree_Uni": DecisionTreeClassifier(ccp_alpha= 0.015, max_depth = 5, min_samples_split= 2),
    "DecTree_Bi": DecisionTreeClassifier(ccp_alpha= 0.015, max_depth = 5, min_samples_split= 2),
    "RForest_Uni": RandomForestClassifier(oob_score=True,  max_depth= 5, min_samples_split= 5, n_estimators= 300),
    "RmForest_Bi": RandomForestClassifier(oob_score=True, max_depth= 10, min_samples_split= 10, n_estimators= 200)
}

# Set Bonferroni-corrected significance level, as it helps us with false positives and robustness, as explained in the report
alpha = 0.05
num_comparisons = len(models) * (len(models) - 1) / 2
adjusted_alpha = alpha / num_comparisons

matrixes = []
#We perform the test multiple times to take randomness out
for i in range(n_tests):
    predictions = {}
    for model_name, model in models.items():
        X_train = X_train_unigram if "Uni" in model_name else X_train_bigram
        X_test = X_test_unigram if "Uni" in model_name else X_test_bigram

        model.fit(X_train, y_train)
        predictions[model_name] = model.predict(X_test)

    # Perform McNemar's test on each model pair and save it in a df for computing the mean and then visualitation
    p_value_matrix = pd.DataFrame(np.nan, index=models.keys(), columns=models.keys())
    results = []
    for (model_1, pred_1), (model_2, pred_2) in itertools.combinations(predictions.items(), 2):
        contingency_table = np.array([
            np.array([(pred_1 == y_test) & (pred_2 == y_test)]).sum(),
            np.array([(pred_1 != y_test) & (pred_2 == y_test)]).sum(),
            np.array([(pred_1 == y_test) & (pred_2 != y_test)]).sum(),
            np.array([(pred_1 != y_test) & (pred_2 != y_test)]).sum()
        ]).reshape(2, 2)

        # Run McNemar's test
        result = mcnemar(contingency_table, exact=True)

        p_value_matrix.loc[model_1, model_2] = result.pvalue
        p_value_matrix.loc[model_2, model_1] = result.pvalue
    
    matrixes.append(p_value_matrix)

p_value_final =  sum(matrixes)/len(matrixes)

if graph_res == True:
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_value_final, annot=True, fmt=".6f", cmap="Greens_r", cbar_kws={'label': 'p-value'})
    plt.title("McNemar Test p-Value Matrix for Model Comparisons")
    plt.tight_layout()

    plt.savefig("NemarResult.png")
        
    mask = p_value_final >= adjusted_alpha
    
    # Create a custom color map (only shows green for significant values)
    cmap = sns.color_palette(["green"])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_value_final, mask=mask, annot=True, fmt=".6f", cmap=cmap, cbar=False,
                linewidths=.5, linecolor="gray")
    plt.title("McNemar Test Significance Matrix (Green = p < Adjusted Alpha)")
    plt.tight_layout()
    plt.savefig("NemarResult_masked.png")
