from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from read_data import X_train, y_train, X_test, y_test
from read_data import X_train_bigram, X_test_bigram

random_forest = RandomForestClassifier(oob_score=True, random_state=42)

parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5,10, 15, 20, 40],
    'min_samples_split': [2, 5, 10, 20],
}

grid_search = GridSearchCV(random_forest, parameters, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)


y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Random Forest Model performance:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# With bigrams

grid_search = GridSearchCV(random_forest, parameters, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_bigram, y_train)

best_model = grid_search.best_estimator_
print("Best parameters (with bigrams):", grid_search.best_params_)


y_pred = best_model.predict(X_test_bigram)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Random Forest Model performance (with bigrams):')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

