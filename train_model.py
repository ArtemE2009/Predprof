import os
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import pickle

stock_names = [
    'TSLA', 'AMD', 'NVDA', 'AAPL', 'INTC', 'LTHM',
    'VZ', 'SMCI', 'MSTR', 'IBM', 'MSFT', 'DELL',
    'HPQ', 'GOOG', 'CSCO', 'AMZN', 'ORCL', 'ADBE',
    'SAP', 'CRM', 'NFLX', 'PYPL', 'QCOM', 'EBAY',
    'RXT', 'DBX', 'DOCU', 'FI', 'ISRG', 'IQ'
]
if not os.path.exists('MODEL'):
    os.makedirs('MODEL')

for stock_name in stock_names:
    quotes_data = pd.read_csv('gross.csv')

    ticker_columns = [col for col in quotes_data.columns if stock_name in col]
    stock_quotes = quotes_data.loc[:, ticker_columns]
    close_column = f"{stock_name} Close"
    current_close = stock_quotes[close_column].iloc[-1]
    quotes_data = pd.read_csv('gross.csv')
    prefix = stock_name

    required_columns = [
        'Date',
        f"{prefix} Open",
        f"{prefix} High",
        f"{prefix} Low",
        f"{prefix} Close",
        f"{prefix} Volume"
    ]

    if os.path.exists('pr.csv'):
        os.remove('pr.csv')
    last_900 = quotes_data.loc[:, required_columns]

    rename_columns = {
        f"Date": "date",
        f"{prefix} Open": "open",
        f"{prefix} High": "high",
        f"{prefix} Low": "low",
        f"{prefix} Close": "close",
        f"{prefix} Volume": "volume"
    }

    last_900.rename(columns=rename_columns, inplace=True)
    last_900.to_csv('pr.csv', index=False)
    df = pd.read_csv('pr.csv')
    df.dropna(inplace=True)
    last_row = df.iloc[-1:]
    last_row.to_csv('prediction.csv', index=False)
    df['pred'] = df['close']
    df['pred'] = df['pred'].shift(-1)
    df.dropna(subset=['pred'], inplace=True)
    df.to_csv('pr_pred.csv', index=False)
    # НЕЙРОСЕТЬ
    df = pd.read_csv('pr_pred.csv')
    df = df.dropna()

    X = df.iloc[:, 1:-1]
    y = df['pred']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    pred_model = RandomForestRegressor(n_estimators=100,
                                       max_depth=15,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       bootstrap=True,
                                       oob_score=True,
                                       n_jobs=-1,
                                       random_state=2,
                                       warm_start=True,
                                       ccp_alpha=0.0)
    model = AdaBoostRegressor(base_estimator=pred_model, n_estimators=25, random_state=5)

    model.fit(X_train, y_train)
    model_path = os.path.join('MODEL', f'{stock_name}_model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    data_pred = pd.read_csv('prediction.csv').iloc[:, 1:]
    stock_prediction = model.predict(data_pred)
    df_original = pd.read_csv('pr_pred.csv')
    stock_past = df_original['close'].iloc[-1]

    test_predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)

    test_rmse = sqrt(mean_squared_error(y_test, test_predictions))
    train_rmse = sqrt(mean_squared_error(y_train, train_predictions))
    print(f"Последняя {prefix} известная цена: {stock_past}")
    print(f"Предсказание {prefix} следующей цены: {stock_prediction}")
    print(f"RMSE на тестовых данных: {test_rmse:.2f}")
    print(f"RMSE на обучающих данных: {train_rmse:.2f}")