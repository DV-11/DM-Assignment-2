from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from read_data import X_train, y_train, X_test, y_test
from read_data import X_train_bigram, X_test_bigram

multinomial_nb = MultinomialNB()

parameters = {
    'alpha': [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
}

grid_search = GridSearchCV(multinomial_nb, parameters, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Multinomial Naive Bayes Model performance:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# With bigrams

grid_search = GridSearchCV(multinomial_nb, parameters, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_bigram, y_train)

best_model = grid_search.best_estimator_
print("Best parameters (with bigrams):", grid_search.best_params_)

y_pred = best_model.predict(X_test_bigram)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Multinomial Naive Bayes Model performance (with bigrams):')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')