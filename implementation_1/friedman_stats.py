from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import friedmanchisquare
import numpy as np
import matplotlib.pyplot as plt

from read_data import X_train, y_train, X_test, y_test
from read_data import X_train_bigram, X_test_bigram

# Prepare unigram data
X_train_unigram, X_test_unigram = X_train, X_test

models_used = {
    "NB_Uni": MultinomialNB(alpha= 0.25),
    "NB_Bi": MultinomialNB(alpha= 0.1),
    "LogReg_Uni": LogisticRegression(C= 25, penalty='l1', solver='liblinear'),
    "LogReg_Bi": LogisticRegression(C= 100, penalty='l1', solver='liblinear'),
    "DecTree_Uni": DecisionTreeClassifier(ccp_alpha= 0.015, max_depth = 5, min_samples_split= 2),
    "DecTree_Bi": DecisionTreeClassifier(ccp_alpha= 0.015, max_depth = 5, min_samples_split= 2),
    "RForest_Uni": RandomForestClassifier(oob_score=True,  max_depth= 5, min_samples_split= 5, n_estimators= 300),
    "RmForest_Bi": RandomForestClassifier(oob_score=True, max_depth= 10, min_samples_split= 10, n_estimators= 200)
}


accuracies_list = []
for name_model, model_implementation in models_used.items():
    X_train = X_train_unigram if "Uni" in name_model else X_train_bigram
    X_test = X_test_unigram if "Uni" in name_model else X_test_bigram
    model_implementation.fit(X_train, y_train)
    y_pred = model_implementation.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_list.append(accuracy)

accuracies_list = np.array(accuracies_list).reshape(-1, len(models_used))

# Run Friedman test
stat, p_value = friedmanchisquare(*accuracies_list.T)
print(f"Friedman statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    #We would do nemenyi or some kind of post-hoc test here, but its not the case, by a lot
    print("There is a significant difference found across models")
else:
    print("No significant differences found across models in Friedman test.")
