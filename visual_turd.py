import tkinter as tk
from tkinter import ttk
import yfinance as yf
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from tkinter import Tk, Label, Text, END, PhotoImage
from PIL import Image, ImageTk
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import pickle


global bg_image

main_background = '#1E213D'
main_foreground = '#FAEBD7'
main_button_background = 0
main_button_active_background = 0
main_button_relief_type = "raised"
main_button_border_width = 10


# #### ФАЙЛЫ ПРОГРАММЫ
# #### MODEL Папка с обученными моделями 30 акций
# #### PR_PRED.CSV ВРЕМЕНЫЙ ФАЙЛ ДЛЯ ОБУЧЕНИЯ НЕЙРОСЕТИ
# #### PREDICTION.CSV ВРЕМЕНЫЙ ФАЙЛ ДЛЯ ОБУЧЕНИЯ НЕЙРОСЕТИ
# #### FOTO.JPG ФОТО ЗАСТАВКИ ОСНОВНОГО ОКНА
# #### TIK.CSV ФАЙЛ С ТИККЕРАМИ S&P500
# #### TRAIN_MODEL.PY ФАЙЛ ОБУЧЕНИЯ 30 МОДЕЛЕЙ ДЛЯ ПОРТФЕЛЯ
# #### GROSS.CSV ВРЕМЕННЫЙ ФАЙЛ В КОТОРЫЙ ЗАГРУЖАЮТСЯ ДАННЫЕ ИЗ ИНТЕРНЕТА
# #### GROSS_R.CSV КОПИЯ ФАЙЛА GROSS.CSV С ПОДГОТОВЛЕННЫМИ ДАННЫМИ ДЛЯ НЕЙРОСЕТИ
# #### PR.CSV ВРЕМЕНЫЙ ФАЙЛ ДЛЯ ОБУЧЕНИЯ НЕЙРОСЕТИ
# #### PREDICT.CSV ФАЙЛ С ДАННЫМИ С ИСТОРИЕЙ ПРЕДСКАЗАНИЙ НЕЙРОСЕТИ
# #### STOCK.CSV ВРЕМЕННЫЙ ФАЙЛ В КОТОРЫЙ ЗАГРУЖАЮТСЯ ДАННЫЕ ИЗ ИНТЕРНЕТА ПО ВВЕДЕННОМУ ТИКЕРУ ДЛЯ НЕЙРОСЕТИ
# #### TICKER.CSV ВРЕМЕННЫЙ ФАЙЛ С ДАННЫМИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ
# #### VISUAL_TURD.PY  ОСНОВНОЙ КОД ПРОГРАММЫ





## ### ЗАГРУЗКА ДАННЫХ C CАЙТА YAHOO FINANCE И СОХРАНЕНИЕ В GROSS.CSV
def create_gross_csv(tickers, filename):
    data = yf.download(tickers, period='max')

    temp_data = {}

    for ticker in tickers:
        if ticker in data['Close']:
            temp_data[f'{ticker} Close'] = data['Close'][ticker]
        if ticker in data['Open']:
            temp_data[f'{ticker} Open'] = data['Open'][ticker]
        if ticker in data['High']:
            temp_data[f'{ticker} High'] = data['High'][ticker]
        if ticker in data['Low']:
            temp_data[f'{ticker} Low'] = data['Low'][ticker]
        if ticker in data['Volume']:
            temp_data[f'{ticker} Volume'] = data['Volume'][ticker]


    clean_data = pd.DataFrame(temp_data)
    clean_data.index = data.index
    clean_data.to_csv(filename)
    print(f'Котировки акций сохранены в файл {filename}.')



usa_tickers = ['TSLA', 'AMD', 'NVDA', 'AAPL', 'INTC', 'LTHM', 'VZ', 'SMCI', 'MSTR', 'IBM',
               'MSFT', 'DELL', 'HPQ', 'GOOG', 'CSCO', 'AMZN', 'ORCL', 'ADBE', 'SAP', 'CRM',
               'NFLX', 'PYPL', 'QCOM', 'EBAY', 'RXT', 'DBX', 'DOCU', 'FI', 'ISRG', 'IQ']

filename = 'gross.csv'

create_gross_csv(usa_tickers, filename)

#  ### СОЗДАЕМ ОКНО ВЫЗАВАЕМОЕ КНОПКОЙ "БИРЖА"

def open_info_window(stock_name):
    info_window = tk.Toplevel()
    info_window.title(stock_name)
    tk.Label(info_window, text=stock_name, font=("Arial", 50)).pack(pady=20)

    info_window.attributes("-fullscreen", True)
    info_window.config(bg=main_background)
    info_window.resizable(False, False)

# ### СОЗДАЕМ И РАЗМЕЩАЕМ КНОПКИ С ТИККЕРАМИ АКЦИЙ В ОКНЕ АКЦИИ

def place_stock_buttons(window, stock_names):
    rows, cols = 5, 6
    for index, stock_name in enumerate(stock_names):
        row, col = divmod(index, cols)
        btn = tk.Button(
            window, text=stock_name, background='#6600FF', foreground=main_foreground, activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 40 ), borderwidth=main_button_border_width, relief=main_button_relief_type, padx=5, pady=10,
            command=lambda sn=stock_name: open_info_window(sn)
        )
        btn.grid(row=row, column=col, sticky='ewns', padx=5, pady=5)

    for column_index in range(cols):
        window.grid_columnconfigure(column_index, weight=1)
    for row_index in range(rows):
        window.grid_rowconfigure(row_index, weight=1)



def fill_text_widget_from_csv(text_widget, filepath):
    data = pd.read_csv(filepath)
    data = data.round(2)
    data_string = data.to_string(index=False, header=True, col_space=25)
    text_widget.delete('1.0', tk.END)
    text_widget.insert(tk.END, data_string)

def create_stock1_window(stock_past, stock_prediction):
    root = tk.Tk()
    root.withdraw()

    stock1_window = tk.Toplevel()
    stock1_window.title("БИРЖА")
    stock1_window.configure(bg='#1D1E33')
    stock1_window.attributes("-fullscreen", True)

    for i in range(6):
        stock1_window.columnconfigure(i, weight=1)
    for i in range(6):
        stock1_window.rowconfigure(i, weight=1)

    exit_button = tk.Button(stock1_window, text="ВЫХОД",background='#6600FF', foreground='#FAEBD7', activebackground='#FF7F50', activeforeground='#FAEBD7', font=('Arial', 20 ), borderwidth=5, relief="groove", padx=5, pady=10, command=stock1_window.destroy)
    exit_button.grid(row=6, column=2, rowspan=1, columnspan=2, sticky="nsew")

    st9_label = tk.Label(stock1_window, text="ИСТРОРИЧЕСКИЕ ДАННЫЕ О КУРСЕ", font=("Arial", 10), bg=main_background,
             fg=main_foreground, wraplength=1100,
             justify="left")
    st9_label.grid(row=3, column=1, rowspan=1, columnspan=4, sticky="nsew")

    data_frame = tk.Frame(stock1_window)
    data_frame.grid(row=4, column=1, rowspan=2, columnspan=4, sticky="ew")

    scroll_y = tk.Scrollbar(data_frame, orient='vertical')
    scroll_y.pack(fill='y', side='right')

    text_widget = tk.Text(data_frame, yscrollcommand=scroll_y.set, wrap="none", font=('Arial', 15))
    text_widget.pack(expand=True, fill='both')

    scroll_y.config(command=text_widget.yview)
    fill_text_widget_from_csv(text_widget, 'stock.csv')


    st6_label = tk.Label(stock1_window, text="ПОСЛЕДНЯЯ ЦЕНА", font=("Arial", 25), bg=main_background,
             fg=main_foreground, wraplength=1100,
             justify="left")
    st6_label.grid(row=1, column=1, rowspan=1, columnspan=1, sticky="nsew")

    st4_label = tk.Label(stock1_window, text=f"{stock_past:.2f}", font=("Arial", 25), bg=main_background,
                         fg=main_foreground, wraplength=1100,
                         justify="left")
    st4_label.grid(row=1, column=2, rowspan=1, columnspan=1, sticky="nsew")


    st7_label = tk.Label(stock1_window, text="ПРОГНОЗ", font=("Arial", 25), bg=main_background,
             fg=main_foreground, wraplength=1100,
             justify="left")
    st7_label.grid(row=1, column=3, rowspan=1, columnspan=1, sticky="nsew")
    prediction_label_color = '#37f734' if stock_prediction > stock_past else 'red'
    formatted_predictions = ", ".join(f"{pred:.2f}" for pred in stock_prediction)
    st5_label = tk.Label(stock1_window, text=formatted_predictions, font=("Arial", 25), bg=main_background,
                         fg=prediction_label_color, wraplength=1100,
                         justify="left")
    st5_label.grid(row=1, column=4, rowspan=1, columnspan=1, sticky="nsew")


    stock1_window.mainloop()


def import_tickers(file_name):
    df = pd.read_csv(file_name)
    return df['Ticker'].tolist()


def create_scrollable_listbox(window, items):
    frame = tk.Frame(window)
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, background='#6600FF', foreground='#FAEBD7',
                         font=('Arial', 20), borderwidth=5)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    for item in items:
        listbox.insert(tk.END, item)

    return frame, listbox

def on_stock_button_click(tk_listbox):
    selected_indices = tk_listbox.curselection()
    if selected_indices:
        selected_index = selected_indices[0]
        ticker = tk_listbox.get(selected_index)

    print("Загружаем данные для тикера:", ticker)
    popup_window = tk.Toplevel()
    popup_window.title("Ожидайте")
    popup_window.attributes("-fullscreen", True)

    popup_label = tk.Label(
        popup_window,
        text="Ожидайте, данные загружаются...",
        font=("Arial", 30),
        bg=main_background,
        fg=main_foreground
    )
    popup_label.pack(expand=True)


    original_img = Image.open("foto.jpg")
    screen_width = popup_window.winfo_screenwidth()
    screen_height = popup_window.winfo_screenheight()

    resized_img = original_img.resize((screen_width, screen_height))
    resized_img.putalpha(150)

    bg_image = ImageTk.PhotoImage(resized_img)


    background_label = tk.Label(popup_window, image=bg_image)
    background_label.place(relwidth=1, relheight=1)

    background_label.image = bg_image
    popup_label = tk.Label(
        popup_window,
        text="Ожидайте, данные загружаются...",
        font=("Arial", 30),
        bg=main_background,
        fg=main_foreground
    )
    popup_label.pack(expand=True)
    popup_window.update_idletasks()
    popup_window.update()


    data = yf.download(ticker, period='max')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data.reset_index(inplace=True)
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    data.to_csv('stock.csv',
                index=False)
    print(f"Котировки акций для {ticker} сохранены в файл 'stock.csv")


    df = pd.read_csv('stock.csv')
    last_row = df.iloc[-1:]
    df.dropna(inplace=True)
    last_row.to_csv('prediction.csv', index=False)
    df['pred'] = df['close']
    df['pred'] = df['pred'].shift(-1)
    df.dropna(subset=['pred'], inplace=True)
    df.to_csv('pr_pred.csv', index=False)
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
    data_pred = pd.read_csv('prediction.csv').iloc[:, 1:]
    stock_prediction = model.predict(data_pred)

    df_original = pd.read_csv('pr_pred.csv')
    stock_past = df_original['close'].iloc[-1]

    test_predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)

    test_rmse = sqrt(mean_squared_error(y_test, test_predictions))
    train_rmse = sqrt(mean_squared_error(y_train, train_predictions))
    print(f"Последняя известная цена: {stock_past}")
    print(f"Предсказание следующей цены: {stock_prediction}")
    print(f"RMSE на тестовых данных: {test_rmse:.2f}")
    print(f"RMSE на обучающих данных: {train_rmse:.2f}")

    popup_window.destroy()

    root = tk.Tk()
    root.withdraw()
    create_stock1_window(stock_past, stock_prediction)


def create_stock_window():
    root = tk.Tk()
    root.withdraw()

    stock_window = tk.Toplevel()
    stock_window.title("БИРЖА")
    stock_window.configure(bg='#1D1E33')
    stock_window.attributes("-fullscreen", True)

    for i in range(4):
        stock_window.columnconfigure(i, weight=1)
    for i in range(5):
        stock_window.rowconfigure(i, weight=1)
    ticker_list = import_tickers('tik.csv')
    st1_label = tk.Label(stock_window, text="ВВЕДИТЕ ТИКЕР", font=("Arial", 10), bg=main_background,
             fg=main_foreground, wraplength=1100,
             justify="left")
    st1_label.grid(row=0, column=1, rowspan=1, columnspan=1, sticky="nsew")
    st2_label =tk.Label(stock_window, text="ПОЛУЧИТЬ ДАННЫЕ", font=("Arial", 10), bg=main_background,
             fg=main_foreground, wraplength=1100,
             justify="left")
    st2_label.grid(row=0, column=2, rowspan=1, columnspan=1, sticky="nsew")

    scrollable_listbox, tk_listbox = create_scrollable_listbox(stock_window, ticker_list)
    scrollable_listbox.grid(row=1, column=1, rowspan=1, columnspan=2, sticky="nsew")

    stock_button = tk.Button(stock_window, text="ПОИСК", background='#6600FF', foreground='#FAEBD7', activebackground='#FF7F50', activeforeground='#FAEBD7', font=('Arial', 30 ), borderwidth=5, relief="groove", padx=5, pady=10,
                             command=lambda: on_stock_button_click(tk_listbox))
    stock_button.grid(row=1, column=2, rowspan=1, columnspan=1, sticky="nsew")

    exit_button = tk.Button(stock_window, text="ВЫХОД",background='#6600FF', foreground='#FAEBD7', activebackground='#FF7F50', activeforeground='#FAEBD7', font=('Arial', 30 ), borderwidth=5, relief="groove", padx=5, pady=10, command=stock_window.destroy)
    exit_button.grid(row=3, column=1, rowspan=1, columnspan=2, sticky="nsew")


    stock_window.mainloop()



### ОКНО ВЫЗЫВАЕМОЕ НАЖАТИЕМ КНОПКИ С ТИККЕРОМ АКЦИИ. \\\ ВЫБИРАЕМ ИНФОРМАЦИЮ ПО НУЖНОМУ ТИКЕРУ АКЦИИ ДЛЯ РАЗМЕЩЕНИЯ В ОКНЕ. ПОДГОТАВЛИВАЕМ ДАННЫЕ ДЛЯ ОБУЧЕНИЯ НЕЙРОСЕТИ
### ### ОБУЧАЕМ НЕЙРОСЕТЬ НА ДАННЫХ НУЖНОГО ТИКЕРА, ДЕЛАЕМ ПРОГНОЗ ПО ДАННЫМ ПОСЛЕДНЕЙ ИЗВЕСТНОЙ ДАТЫ
### ### ### ПОДГОТАВЛИВАЕМ ДАННЫЕ ДЛЯ ПОСТРОЕНИЯ СВЕЧНОГО ГРАФИКА
#### СОЗДАЕМ ОКНО И РАЗМЕЩАЕМ В НЕМ ИНФОРМАЦИЮ ПО ТИККЕРУ (НАЗВАНИЕ ТИККЕРА, НАЗВАНИЕ КОМПАНИИ, ПОСЛЕДНЯЯ ЦЕНА АКЦИИ, ПРОГНОЗ ЦЕНЫ ПО АКЦИИ, ИНФОРМАЦИЯ О КОМПАНИИ, СВЕЧНОЙ ГРАФИК)

def open_info_window(stock_name):

    popup_window = tk.Toplevel()
    popup_window.title("Ожидайте")
    popup_window.attributes("-fullscreen", True)

    popup_label = tk.Label(
        popup_window,
        text="Ожидайте, данные загружаются...",
        font=("Arial", 30),
        bg=main_background,
        fg=main_foreground
    )
    popup_label.pack(expand=True)

    original_img = Image.open("foto.jpg")
    screen_width = popup_window.winfo_screenwidth()
    screen_height = popup_window.winfo_screenheight()

    resized_img = original_img.resize((screen_width, screen_height))
    resized_img.putalpha(150)

    bg_image = ImageTk.PhotoImage(resized_img)

    background_label = tk.Label(popup_window, image=bg_image)
    background_label.place(relwidth=1, relheight=1)

    background_label.image = bg_image
    popup_label = tk.Label(
        popup_window,
        text="Ожидайте, данные загружаются...",
        font=("Arial", 30),
        bg=main_background,
        fg=main_foreground
    )
    popup_label.pack(expand=True)
    popup_window.update_idletasks()
    popup_window.update()


    text_data = pd.read_csv('list.csv', header=None, sep=';', names=['ticker', 'name', 'description'])
    stock_info = text_data[text_data['ticker'] == stock_name]

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
    df.to_csv('pr.csv', index=False)
    last_row.to_csv('prediction.csv', index=False)
    df['pred'] = df['close']
    df['pred'] = df['pred'].shift(-1)
    df.dropna(subset=['pred'], inplace=True)
    df.to_csv('pr_pred.csv', index=False)

    model_path = f'MODEL/{stock_name}_model.pkl'

    with open(model_path, 'rb') as file:
            model = pickle.load(file)

    data_pred = pd.read_csv('prediction.csv').iloc[:, 1:]

    prediction = model.predict(data_pred)

    df_original = pd.read_csv('pr_pred.csv')
    past = df_original['close'].iloc[-1]

    print(f"Последняя известная цена: {past}")
    print(f"Предсказание следующей цены: {prediction}")


### ### ДАННЫЕ ДЛЯ ГРАФИКА
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

    if os.path.exists('ticker.csv'):
        os.remove('ticker.csv')

    last_90 = quotes_data.loc[:, required_columns].tail(90)

    rename_columns = {
        f"Date": "date",
        f"{prefix} Open": "open",
        f"{prefix} High": "high",
        f"{prefix} Low": "low",
        f"{prefix} Close": "close",
        f"{prefix} Volume": "volume"
    }

    last_90.rename(columns=rename_columns, inplace=True)
    last_90.to_csv('ticker.csv', index=False)
    print("Файл 'ticker.csv' успешно создан.")

    plt_data = pd.read_csv('ticker.csv')
    plt_data['date'] = pd.to_datetime(plt_data['date'])

### ### СОХРАНЯЕМ ДАННЫЕ О СДЕЛАННЫХ ПРЕДСКАЗАНИЯХ
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_data = pd.DataFrame([[current_date, stock_name, prediction]], columns=['Дата', 'Тикер', 'Прогноз'])
    existing_data = pd.read_csv('predict.csv')
    updated_data = pd.concat([new_data, existing_data]).reset_index(drop=True)
    updated_data.to_csv('predict.csv', mode='w', index=False, header=True)
    popup_window.destroy()


#### СОЗДАЕМ ОКНО
    info_window = tk.Toplevel()
    info_window.title(stock_name)
    info_window.geometry("1280x720")
    info_window.configure(bg=main_background)
    info_window.attributes("-fullscreen", True)
    info_window.resizable(False, False)
    for i in range(3):
        info_window.columnconfigure(i, weight=1)
    for i in range(9):
        info_window.rowconfigure(i, weight=1)


    info1_label = tk.Label(info_window, text=f"{stock_name}      {stock_info.iloc[-1]['name']}", font=("Arial", 20),
                           bg=main_background, fg=main_foreground)
    info1_label.grid(row=0, column=1, rowspan=1, columnspan=1, sticky="nsew", pady=20)

    prediction_label_color = '#37f734' if prediction > current_close else 'red'
    predicted_value = prediction[0]
    info2_label = tk.Label(info_window, text=f"Цена {past:.2f}           Прогноз на завтра {predicted_value:.2f}",
                           font=("Arial", 18), fg=prediction_label_color, bg=main_background)
    info2_label.grid(row=1, column=1, rowspan=1, columnspan=1, sticky="nsew")

    info3_label = tk.Label(info_window, text=stock_info.iloc[-1]['description'], font=("Arial", 10), bg=main_background,
                           fg=main_foreground, wraplength=1100, justify="left")
    info3_label.grid(row=2, column=0, rowspan=2, columnspan=3, sticky="nsew")

    #### СТРОИМ ГРАФИК И РАЗМЕЩАЕМ ЕГО В ОКНЕ
    ohlc = plt_data.loc[:, ['date', 'open', 'high', 'low', 'close', 'volume']]
    ohlc['date'] = ohlc['date'].apply(mdates.date2num)

    fig, ax = plt.subplots()

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate()

    candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='g', colordown='r')

    ax.set_xlabel('Дата', fontsize=10)
    ax.set_ylabel('Цена', fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    fig.set_size_inches(12, 4, forward=True)

    canvas = FigureCanvasTkAgg(fig, master=info_window)
    canvas.draw()

    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=5, column=1, rowspan=1, columnspan=1, sticky="nsew")

    info4_label = tk.Label(info_window, text="           ", font=("Arial", 24), bg=main_background, fg=main_foreground)
    info4_label.grid(row=7, column=1, rowspan=1, columnspan=1, sticky="nsew")

    menu_button = tk.Button(info_window, text="ВЫХОД", background='#6600FF', foreground=main_foreground, activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 20), borderwidth=main_button_border_width, relief=main_button_relief_type, padx=5, pady=10, command=lambda: [info_window.destroy()])

    menu_button.grid(row=8, column=1, rowspan=1, columnspan=1, sticky="nsew")

    info_window.mainloop()


#### ОКНО С КНОПКАМИ ТИККЕРОВ.

def create_main_window1(window_to_close, title):

    main_window1 = tk.Tk()
    main_window1.title(title)

    main_window1.config(bg=main_background)
    main_window1.attributes("-fullscreen", True)
    main_window1.resizable(False, False)

    # Список тикеров для создания кнопок
    stock_names = [
        'TSLA', 'AMD', 'NVDA', 'AAPL', 'INTC', 'LTHM',
        'VZ', 'SMCI', 'MSTR', 'IBM', 'MSFT', 'DELL',
        'HPQ', 'GOOG', 'CSCO', 'AMZN', 'ORCL', 'ADBE',
        'SAP', 'CRM', 'NFLX', 'PYPL', 'QCOM', 'EBAY',
        'RXT', 'DBX', 'DOCU', 'FI', 'ISRG', 'IQ'
    ]

    place_stock_buttons(main_window1, stock_names)

    exit_button_frame = tk.Frame(main_window1)
    exit_button_frame.grid(row=5, column=5, sticky='ewns')
    exit_button = tk.Button(exit_button_frame,
                            text="Выход", background='#6600FF', foreground=main_foreground, activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 20 ), borderwidth=main_button_border_width, relief=main_button_relief_type, padx=5, pady=10,
                            command=lambda: [main_window1.destroy()])
    exit_button.pack(fill='x')

    main_window1.mainloop()



### #### ОКНО ПОДТВЕРЖДЕНИЯ ВЫХОДА ИЗ ПРОГРАММЫ С КНОПКАМИ ВЫХОД И ОТМЕНА

def ask_exit(main_window):
    confirm_window = tk.Toplevel(main_window, bg=main_background)
    confirm_window.title("Выход")
    confirm_window.geometry("400x250")
    confirm_window.transient(main_window)

    label_confirm = tk.Label(confirm_window, text="Вы уверены, что хотите выйти?", bg=main_background, fg=main_foreground)
    label_confirm.pack(pady=10)

    btn_exit = tk.Button(confirm_window, text="ВЫЙТИ", background='#6600FF', foreground=main_foreground, activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 10), borderwidth=3, relief=main_button_relief_type, padx=5, pady=10,
                         command=lambda: main_window.destroy())
    btn_exit.pack(side='left', fill='x', expand=True, padx=5, pady=5)

    btn_cancel = tk.Button(confirm_window, text="ОТМЕНА", background='#6600FF', foreground=main_foreground, activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 10), borderwidth=3, relief=main_button_relief_type, padx=5, pady=10,
                           command=confirm_window.destroy)
    btn_cancel.pack(side='right', fill='x', expand=True, padx=5, pady=5)


#### #### ОКНО С ИСТОРИЕЙ ПРЕДСКАЗАНИЙ

def create_main_window2(parent, title):
    new_window = tk.Toplevel(parent)
    new_window.title(title)
    new_window.attributes("-fullscreen", True)
    new_window.resizable(False, False)
    style = ttk.Style()

    style.configure("Treeview", font=('Arial', 14))
    style.configure("Treeview.Heading", font=('Arial', 16))
    style.configure("Vertical.TScrollbar", arrowcolor='blue')

    tree_frame = tk.Frame(new_window)
    tree_frame.pack(fill='both', expand=True)

    tree_scroll = tk.Scrollbar(tree_frame)
    tree_scroll.pack(side='right', fill='y')

    tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, selectmode="extended")
    tree.pack(fill='both', expand=True)

    tree_scroll.config(command=tree.yview)
    tree['columns'] = ("Date", "Price", "Prediction")

    tree.column("#0", width=0, stretch=tk.NO)
    for col in tree['columns']:
        tree.column(col, anchor=tk.CENTER, width=80)

    for col in tree['columns']:
        tree.heading(col, text=col, anchor=tk.CENTER)

    data = pd.read_csv("predict.csv")
    # Вставка данных в Treeview
    for i in data.index:
        tree.insert("", "end", values=list(data.loc[i]))

    exit_button = tk.Button(new_window, text="ВЫХОД", background='#6600FF', foreground=main_foreground, activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 20 ), borderwidth=main_button_border_width, relief=main_button_relief_type, padx=5, pady=10, command=new_window.destroy)
    exit_button.pack(fill='x', side='bottom')


#### #### ОСНОВНОЕ ОКНО ПРОГРАММЫ

def create_main_window():
    global bg_image
    window = tk.Tk()
    window.title("AI STOCK")
    window.attributes("-fullscreen", True)
    window.resizable(False, False)  # Открываем окно на весь экран
    window.configure(bg=main_background)
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    window.geometry(f"{screen_width}x{screen_height}+0+0")

    if 'bg_image' not in globals():
        original_img = Image.open("foto.jpg")
        resized_img = original_img.resize((screen_width, screen_height))
        bg_image = ImageTk.PhotoImage(resized_img)


    background_label = Label(window, image=bg_image)
    background_label.place(relwidth=1, relheight=1)
    background_label.image = bg_image

    title0_label = tk.Label(window, text="", bg=main_background, fg=main_foreground, font=("Arial", 20))
    title0_label.pack(pady=10)
    title_label = tk.Label(window, text="AI STOCK", bg=main_background, fg=main_foreground, font=("Arial", 80))
    title_label.pack(pady=10)

    text_block = tk.Text(window, bg=main_background, fg=main_foreground, font=("Arial", 12))
    text_block.pack(expand=True)
    text_block.tag_configure("center", justify='center')
    text_block.insert(tk.END,
f"""

   Приложение AI STOCK работает на базе искусственного интеллекта и
   позволяет спрогнозировать стоимость акций ведущих высокотехнологичных компаний.

   Вам доступны:
     - прогноз цены акций из Вашего "Потфеля"
     - историческая справка по динамике котировок 30 акций высокотехнологического сектора из Вашего "Портфеля"
     - информация о компаниях из Вашего портфеля
     - исторические данные о ценах на акции входящие в индекс S&P500
     - последняя цена и прогноз цены акции компании, входящей в индекс S&P500
     - данные о сделанных запросах прогноза цены
   Программа разработана для участия в предпрофессиональной олимпиаде. 
  

Разработчики: 
Ефремцев Артем, 8В1; Университетский лицей №1511 предуниверситария НИЯУ МИФИ 
Косолапов Кирилл, 9И1; Университетский лицей №1511 предуниверситария НИЯУ МИФИ
Баженов Лев, 8В1; Университетский лицей №1511 предуниверситария НИЯУ МИФИ
Ананьев Артëм, 9И1; Университетский лицей №1511 предуниверситария НИЯУ МИФИ

г. Москва. 2024. 
v1.1.1.3
""", "left")
    text_block.config(state="disabled")

    buttons_frame = tk.Frame(window, bg=main_background)
    buttons_frame.pack(fill='x')

    btn_courses = tk.Button(buttons_frame, text="ПОРТФЕЛЬ", background='#6600FF', foreground=main_foreground, activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 20 ), borderwidth=main_button_border_width, relief=main_button_relief_type,
                            command=lambda: create_main_window1(window, "Курсы"))
    btn_courses.pack(fill='x', side='left', expand=True, padx=5, pady=5)

    btn_stock = tk.Button(buttons_frame, text="БИРЖА", background='#6600FF', foreground=main_foreground, activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 20 ), borderwidth=main_button_border_width, relief=main_button_relief_type,
                         command=create_stock_window)
    btn_stock.pack(fill='x', side='left', expand=True, padx=5, pady=5)

    btn_courses = tk.Button(buttons_frame, text="ИСТОРИЯ", background='#6600FF', foreground=main_foreground,
                            activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 20),
                            borderwidth=main_button_border_width, relief=main_button_relief_type,
                            command=lambda: create_main_window2(window, "История"))
    btn_courses.pack(fill='x', side='left', expand=True, padx=5, pady=5)

    btn_exit_main = tk.Button(buttons_frame, text="ВЫХОД", background='#6600FF', foreground=main_foreground,
                              activebackground='#FF7F50', activeforeground=main_foreground, font=('Arial', 20), borderwidth=main_button_border_width,
                              relief=main_button_relief_type,
                              command=lambda: ask_exit(window))
    btn_exit_main.pack(fill='x', side='left', expand=True, padx=5, pady=5)


    window.mainloop()


create_main_window()


#ДЛЯ КНОПОК: , background='#345', foreground='black', font=('Arial', 14 ), borderwidth=3, relief=main_button_relief_type, padx=5, pady=10 , activebackground='#345',activeforeground='white'


