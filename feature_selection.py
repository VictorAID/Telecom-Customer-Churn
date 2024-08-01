from sklearn.feature_selection import SelectKBest

def select_features(X, y):
    selection = SelectKBest()  # k=10 default
    X_new = selection.fit_transform(X, y)
    return X_new

