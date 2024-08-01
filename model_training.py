from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=150,criterion='gini', max_depth=15, min_samples_leaf=10, min_samples_split=6,random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    accuracy = accuracy_score(pred, y_test)
    conf_matrix = confusion_matrix(pred, y_test)
    class_report = classification_report(pred, y_test)
    
    return accuracy, conf_matrix, class_report

