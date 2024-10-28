from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from read_data import X_train, y_train, X_test, y_test
from read_data import X_train_bigram, X_test_bigram

logistic_regression = LogisticRegression(random_state=42, penalty='l1', solver='liblinear')

parameters = {
    'C': [0.5, 1.0, 1.5, 2.0, 2.5],
}

grid_search = GridSearchCV(logistic_regression, parameters, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Linear Regression Model performance:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# With bigrams

grid_search = GridSearchCV(logistic_regression, parameters, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_bigram, y_train)

best_model = grid_search.best_estimator_
print("Best parameters (with bigrams):", grid_search.best_params_)

y_pred = best_model.predict(X_test_bigram)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Linear Regression Model performance (with bigrams):')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')