import xgboost as xgb
import lightgbm as lgb
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import compute_sample_weight

from util import compute_metrics
from constants import TITLE_KEYWORDS


def train(model_name, features, labels, balance_classes=False):

    if model_name in MODEL_DICT:
        model, cv_grid = MODEL_DICT[model_name]()
    else:
        raise ValueError(f'Model {model_name} not found.')
    
    pipeline = Pipeline([('feature_extractor', CustomTFIDF()), ('model', model)])
    # Update the cv_grid by prepending 'model__' to each parameter name
    cv_grid = {f'model__{k}': v for k, v in cv_grid.items()}


    # Grid search for the best parameters
    grid_search = GridSearchCV(pipeline, param_grid=cv_grid, cv=5, verbose=100)
    
    if balance_classes:
        sample_weights = compute_sample_weight('balanced', labels)
        grid_search.fit(features, labels, model__sample_weight=sample_weights)
    else:
        grid_search.fit(features, labels)

    # Get best score
    print(f'Best score: {grid_search.best_score_}')
 
    print('Evaluating on train set...')
    model_name = f'{model_name}_balanced' if balance_classes else model_name
    eval(grid_search, model_name, features, labels, model_name, split='train')

    # Print the best parameters
    print(f'Best parameters: {grid_search.best_params_}')

    # Save the grid search
    if balance_classes:
        joblib.dump(grid_search, f'{model_name}_balanced.joblib')
    else:
        joblib.dump(grid_search, f'{model_name}.joblib')

    return grid_search


def eval(model, model_name, features, labels, split='val'):
    '''
    Make predictions on val/test data and compute metrics
    '''
    predictions = model.predict(features)
    prob_predictions = model.predict_proba(features)
    compute_metrics(labels, predictions, prob_predictions, model_name, split)


def get_decision_tree(use_best_params=True):
    decision_tree = DecisionTreeClassifier()
    cv_grid = {
        'max_depth': [10, 20, 40], 
        'min_samples_split': [2, 4, 8],  
        'min_samples_leaf': [1, 3, 6]
    }
    best_params = {
        'max_depth': [20], 
        'min_samples_split': [4],  
        'min_samples_leaf': [6]
    }
    if use_best_params:
        return decision_tree, best_params
    return decision_tree, cv_grid


def get_random_forest():
    random_forest = RandomForestClassifier()
    cv_grid = {
        'n_estimators': [40], 
        'max_depth': [20], 
        'min_samples_split': [4], 
        'min_samples_leaf': [3]
    }
    return random_forest, cv_grid


def get_gradient_boost():
    gradient_boost = GradientBoostingClassifier()
    cv_grid = {
        'n_estimators': [30], 
        'max_depth': [2, 3, 5],
        'min_samples_split': [2, 4, 8], 
        'min_samples_leaf': [1, 3, 6]
    }
    return gradient_boost, cv_grid


def get_svm():
    svm = SVC(class_weight='balanced')
    cv_grid = {
        # 'kernel': ['linear', 'rbf', 'sigmoid'], 
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100]
    }
    return svm, cv_grid


def get_logistic_regression(use_best_params=False):
    logistic_regression = LogisticRegression(class_weight='balanced')
    cv_grid = {
        'C': [0.1, 1, 10, 100]
    }
    best_params = {
        'C': [0.1]
    }
    if use_best_params:
        return logistic_regression, best_params
    return logistic_regression, cv_grid


def get_xgboost(use_best_params=True):
    xgboost = xgb.XGBClassifier()
    cv_grid = {
        'n_estimators': [40, 60, 80],
        'max_depth': [2, 3, 5],
        'learning_rate': [0.1, 1, 1]
    }
    best_params = {
        'n_estimators': [60],
        'max_depth': [2],
        'learning_rate': [1],
    }
    if use_best_params:
        return xgboost, best_params
    return xgboost, cv_grid


def get_neural_network(use_best_params=True):
    neural_net = MLPClassifier()
    cv_grid = {
        'hidden_layer_sizes': [(100,), (100, 100)]
    }
    best_params = {
        'hidden_layer_sizes': [(100,)]
    }
    if use_best_params:
        return neural_net, best_params
    return neural_net, cv_grid


def get_lightgbm(use_best_params=True):
    lightgbm = lgb.LGBMClassifier()
    cv_grid = {
        'num_leaves': [10, 20, 40], 
        'max_depth': [10, 20, 40], 
        'min_data_in_leaf': [1, 3, 6],
    }
    best_params = {
        'max_depth': [10], 
        'min_data_in_leaf': [6], 
        'num_leaves': [10]
    }
    if use_best_params:
        return lightgbm, best_params
    return lightgbm, cv_grid


class CustomTFIDF(BaseEstimator, TransformerMixin):
    '''
    Custom TF-IDF transformer that uses a vocabulary that is the union of the top 100 features and TITLE_KEYWORDS
    '''
    def __init__(self):
        self.vectorizer = None

    def fit(self, X, y=None):
        vectorizer = TfidfVectorizer(max_features=100)
        vectorizer = vectorizer.fit(X['job_titles'])
        # feature_names = list(vectorizer.get_feature_names_out())
        # vocabulary = list(set(feature_names + TITLE_KEYWORDS))
        # vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        # vectorizer = vectorizer.fit(X['job_titles'])
        self.vectorizer = vectorizer
        return self


    def transform(self, X, y=None):
        job_title_features = self.vectorizer.transform(X['job_titles'])
        job_title_features = pd.DataFrame(job_title_features.toarray(), columns=self.vectorizer.get_feature_names_out())
        X = pd.concat([X.reset_index(drop=True), job_title_features.reset_index(drop=True)], axis=1)
        X = X.drop(columns=['job_titles'])
        return X


MODEL_DICT = {
    'decision_tree': get_decision_tree,
    'random_forest': get_random_forest,
    'logistic_regression': get_logistic_regression,
    'xgboost': get_xgboost,
    'neural_network': get_neural_network,
    'gradient_boost': get_gradient_boost,
    'svm': get_svm,
    'lightgbm': get_lightgbm
}