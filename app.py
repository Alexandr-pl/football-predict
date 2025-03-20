import os
import joblib
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

# Подавление предупреждений
warnings.filterwarnings("ignore", category=UserWarning)

# Инициализация приложения Dash
app = dash.Dash(__name__)
server = app.server

# 1. Загрузка и подготовка данных
try:
    df = pd.read_csv(
        'football_predict.csv',
        on_bad_lines='skip',
        encoding='utf-8',
        engine='python'
    )
    print(f"Успешно загружено строк: {len(df)}")
    
    # Предобработка данных
    df['season'] = df['season'].str.split('/').str[0].astype(int)
    
    # Проверка наличия необходимых колонок
    required_columns = ['league', 'home_team', 'away_team', 'season'] + [
        'home_speed', 'home_creation', 'home_defence', 'home_avg_goals_last_5',
        'away_speed', 'away_creation', 'away_defence', 'away_avg_goals_last_5',
        'defence_diff', 'home_draws_last_5', 'away_draws_last_5', 'outcome'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Отсутствует обязательная колонка: {col}")

except Exception as e:
    print(f"Ошибка загрузки данных: {str(e)}")
    raise

# 2. Кодирование целевой переменной
le = LabelEncoder()
df['outcome'] = le.fit_transform(df['outcome'])

# 3. Загрузка или обучение модели
model_path = 'football_model.pkl'
features = [
    'home_speed', 'home_creation', 'home_defence', 'home_avg_goals_last_5',
    'away_speed', 'away_creation', 'away_defence', 'away_avg_goals_last_5',
    'defence_diff', 'home_draws_last_5', 'away_draws_last_5'
]

model = None
try:
    model = joblib.load(model_path)
    print("Модель успешно загружена из кэша")
except Exception as e:
    print(f"Ошибка загрузки модели: {str(e)}")
  

# Макет приложения
app.layout = html.Div([
    html.H1("Футбольный аналитический дашборд", style={'textAlign': 'center'}),
    
    # Блок выбора лиги
    html.Div([
        dcc.Dropdown(
            id='league-selector',
            options=[{'label': league, 'value': league} for league in sorted(df['league'].unique())],
            placeholder="Выберите лигу",
            style={'width': '80%', 'margin': '20px auto'}
        )
    ]),
    
    # Блок выбора команд
    html.Div([
        dcc.Dropdown(
            id='home-team',
            options=[],
            placeholder="Сначала выберите домашнюю команду",
            style={'width': '45%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='away-team',
            options=[],
            placeholder="Сначала выберите гостевую команду",
            style={'width': '45%', 'display': 'inline-block', 'margin-left': '5%'}
        )
    ], style={'width': '80%', 'margin': '20px auto'}),
    
    # Слайдер сезона
    html.Div([
        html.Label("Выбор сезона:", 
                  style={'font-weight': 'bold', 'margin-bottom': '10px'}),
        dcc.Slider(
            id='season-slider',
            min=int(df['season'].min()),
            max=int(df['season'].max()),
            marks={},
            value=int(df['season'].min()),
            step=1
        )
    ], style={'width': '80%', 'margin': '20px auto'}),
    
    # Графики
    html.Div([
        dcc.Graph(id='comparison-bar-chart')
    ], style={'width': '90%', 'margin': 'auto'}),
    
    html.Div([
        dcc.Graph(id='outcome-prediction'),
        dcc.Graph(id='historical-stats')
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin-top': '30px'})
])

# Callback для обновления фильтров
@app.callback(
    [Output('home-team', 'options'),
     Output('away-team', 'options'),
     Output('season-slider', 'marks'),
     Output('season-slider', 'value')],
    [Input('league-selector', 'value')]
)
def update_filters(selected_league):
    if not selected_league:
        return [], [], {}, int(df['season'].min())
    
    filtered_df = df[df['league'] == selected_league]
    
    teams = [{'label': team, 'value': team} for team in sorted(filtered_df['home_team'].unique())]
    seasons = sorted(filtered_df['season'].unique())
    
    marks = {str(year): {'label': str(year), 'style': {'transform': 'rotate(45deg)'}} 
            for year in seasons}
    
    return teams, teams, marks, seasons[-1] if seasons else int(df['season'].min())

# Основной callback
@app.callback(
    [Output('comparison-bar-chart', 'figure'),
     Output('outcome-prediction', 'figure'),
     Output('historical-stats', 'figure')],
    [Input('home-team', 'value'),
     Input('away-team', 'value'),
     Input('season-slider', 'value'),
     Input('league-selector', 'value')]
)
def update_dashboard(home_team, away_team, selected_season, selected_league):
    try:
        if not all([home_team, away_team, selected_league]):
            return go.Figure(), go.Figure(), go.Figure()
        
        # Фильтрация данных
        filtered_df = df[
            (df['league'] == selected_league) &
            (df['home_team'] == home_team) &
            (df['away_team'] == away_team) &
            (df['season'] == selected_season)
        ]
        
        # Инициализация фигур
        bar_fig = go.Figure()
        pie_fig = go.Figure()
        line_fig = go.Figure()

        # Бар-чарт сравнения
        categories = ['Скорость реализации атаки', 'Создание моментов', 'Агресивность защиты']
        home_values = [
            filtered_df['home_speed'].mean(),
            filtered_df['home_creation'].mean(),
            filtered_df['home_defence'].mean()
        ]
        away_values = [
            filtered_df['away_speed'].mean(),
            filtered_df['away_creation'].mean(),
            filtered_df['away_defence'].mean()
        ]
        
        bar_fig.add_trace(go.Bar(
            x=categories,
            y=home_values,
            name=home_team,
            marker_color='#3498db'
        ))
        bar_fig.add_trace(go.Bar(
            x=categories,
            y=away_values,
            name=away_team,
            marker_color='#e74c3c'
        ))
        bar_fig.update_layout(
            title='Сравнение характеристик команд',
            barmode='group',
            showlegend=True
        )
        
        # Прогноз исхода
        if model is not None and not filtered_df.empty:
            try:
                input_data = filtered_df[features].mean().to_frame().T
                probas = model.predict_proba(input_data)[0]
            except Exception as e:
                print(f"Ошибка прогнозирования: {str(e)}")
                probas = [0.33, 0.33, 0.33]
        else:
            probas = [0.33, 0.33, 0.33]
        
        pie_fig = go.Figure(
            go.Pie(
                labels=le.classes_,
                values=probas,
                hole=0.4,
                marker_colors=['#2ecc71', '#f1c40f', '#e74c3c']
            )
        )
        pie_fig.update_layout(title='Прогноз исхода матча')
        
        # Историческая статистика
        historical_df = df[
            (df['league'] == selected_league) &
            (df['home_team'] == home_team) &
            (df['away_team'] == away_team)
        ]
        
        if not historical_df.empty:
            line_fig = px.line(
                historical_df,
                x='season',
                y='home_team_goal',
                title='История голов домашней команды',
                markers=True,
                labels={'season': 'Сезон', 'home_goal': 'Голы'}
            )
        else:
            line_fig = go.Figure()
            line_fig.add_annotation(text="Нет исторических данных")
        
        return bar_fig, pie_fig, line_fig
    
    except Exception as e:
        print(f"Ошибка обновления дашборда: {str(e)}")
        return go.Figure(), go.Figure(), go.Figure()

if __name__ == '__main__':
    app.run(debug=True)