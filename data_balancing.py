from imblearn.combine import SMOTEENN
from collections import Counter

def balance_data(X_train, y_train):
    st = SMOTEENN()
    X_train_st, y_train_st = st.fit_resample(X_train, y_train)
    return X_train_st, y_train_st

