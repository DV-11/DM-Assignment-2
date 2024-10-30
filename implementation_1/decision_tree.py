from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from read_data import X_train, y_train, X_test, y_test
from read_data import X_train_bigram, X_test_bigram

decision_tree = DecisionTreeClassifier(random_state=42)

parameters = {
    'max_depth': [5, 10, 20, 40],
    'min_samples_split': [2, 5, 10, 20, 30],
    'ccp_alpha': [0.005, 0.01, 0.015]
}

grid_search = GridSearchCV(decision_tree, parameters, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Decision Tree Model performance:')
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')

# with bigrams: 

grid_search = GridSearchCV(decision_tree, parameters, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_bigram, y_train)

best_model = grid_search.best_estimator_
print("Best parameter (with bigrams):", grid_search.best_params_)

y_pred = best_model.predict(X_test_bigram)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Decision Tree Model performance (with bigrams):')
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')