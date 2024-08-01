from data_preprocessing import load_data, preprocess_data
from feature_selection import select_features
from data_balancing import balance_data
from model_training import train_model, evaluate_model
from model_saving import save_model, load_model
from sklearn.model_selection import train_test_split

def main():
    data = load_data('Telco-Customer-Churn.csv')
    X, y = preprocess_data(data)
    X_new = select_features(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2,random_state=42)
    X_train_st, y_train_st = balance_data(X_train, y_train)
    X_train_sap, X_test_sap, y_train_sap, y_test_sap = train_test_split(X_train_st, y_train_st, test_size=0.2,random_state=42)
    model = train_model(X_train_sap, y_train_sap)
    save_model(model, 'Rf_Model.sav')

    loaded_model = load_model('Rf_Model.sav')
    accuracy, conf_matrix, class_report = evaluate_model(loaded_model, X_test_sap, y_test_sap)
    
    print(f'Loaded model accuracy score : {accuracy}')
    print(f'Loaded model confusion matrix :\n {conf_matrix}')
    print(f'Loaded model classification report :\n {class_report}')

if __name__ == "__main__":
    main()
