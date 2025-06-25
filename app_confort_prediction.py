
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Assistant Confort", layout="centered")

# --- Personnalisation UI
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f7fa;
            background-image: linear-gradient(to bottom right, #f5f7fa, #e0ecf8);
        }
        .element-container img {
            display: block;
            margin: 0 auto;
        }
        h1 {
            color: #1f4e79;
        }
    </style>
""", unsafe_allow_html=True)

from PIL import Image
logo = Image.open("Centrale Med.png")
st.image(logo, width=180)


st.title("🏡 Assistant Confort et Consommation Énergétique")
st.markdown("Prédisez automatiquement la consommation énergétique estimée et la température intérieure recommandée selon vos préférences de confort.")

# --- Chargement des données
df_daily = pickle.load(open("df_daily.pickle", "rb"))

@st.cache_data
def get_weather_forecast(lat=43.7, lon=7.25):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,direct_radiation&forecast_days=1&timezone=auto"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        now = pd.Timestamp.now().round("h")
        df_now = df[df["time"] == now]
        if not df_now.empty:
            return df_now["temperature_2m"].values[0], df_now["direct_radiation"].values[0]
    return None, None

# --- Préparation
df = df_daily.rename(columns={
    'TEMP_RATURES_MEAN_BUILDING': 'temp_int',
    'CONSOMMATION_ENERGIE_ELEC': 'conso_energie',
    'TEMP_RATURES_EXT_RIE_TEMPRATURE_EXTRIEU': 'temp_ext',
    'CONSOMMATION_ENERGIE_PV': 'ensoleillement'
}).dropna()
df['jour'] = pd.to_datetime(df.index).dayofweek

q33 = df['conso_energie'].quantile(0.33)
q66 = df['conso_energie'].quantile(0.66)
low_df = df[df['conso_energie'] < q33]
med_df = df[(df['conso_energie'] >= q33) & (df['conso_energie'] <= q66)]
high_df = df[df['conso_energie'] > q66]

# --- UI météo
with st.expander("🌤️ Utiliser la météo en direct ou saisir manuellement :"):
    use_api = st.radio("Source des données météo", options=["API météo", "Saisie manuelle"], index=0)

if use_api == "API météo":
    temp_ext, ensoleillement = get_weather_forecast()
    if temp_ext is None:
        st.warning("⚠️ Données météo indisponibles pour le moment.")
        st.stop()
    st.success(f"🌡️ Température extérieure actuelle : {temp_ext} °C")
    st.success(f"🔆 Ensoleillement actuel : {ensoleillement} W/m²")
else:
    temp_ext = st.number_input("🌡️ Température extérieure (°C)", value=5.0, step=0.5)
    ensoleillement = st.number_input("🔆 Ensoleillement (W/m²)", value=30.0, step=10.0)

# --- Aperçu des données
st.subheader("📊 Échantillon de données historiques")
st.dataframe(df[['temp_ext', 'ensoleillement', 'conso_energie', 'temp_int']].head())

st.subheader("📈 Statistiques générales")
st.dataframe(df[['temp_ext', 'ensoleillement', 'conso_energie', 'temp_int']].describe().T)

# --- Segmentation
st.subheader("📉 Plages de consommation par segment")
col1, col2, col3 = st.columns(3)
col1.metric("🔋 Low", f"{low_df['conso_energie'].min():.2f} – {low_df['conso_energie'].max():.2f} kWh")
col2.metric("⚖️ Medium", f"{med_df['conso_energie'].min():.2f} – {med_df['conso_energie'].max():.2f} kWh")
col3.metric("🔥 High", f"{high_df['conso_energie'].min():.2f} – {high_df['conso_energie'].max():.2f} kWh")

# --- Visualisations
with st.expander("🔎 Visualisations avancées"):
    st.subheader("📦 Boxplot consommation par segment")
    df_plot = pd.DataFrame({
        'Conso (kWh)': pd.concat([low_df['conso_energie'], med_df['conso_energie'], high_df['conso_energie']]),
        'Segment': ['Low']*len(low_df) + ['Medium']*len(med_df) + ['High']*len(high_df)
    })
    fig, ax = plt.subplots()
    sns.boxplot(x='Segment', y='Conso (kWh)', data=df_plot, hue='Segment', palette='Set2', dodge=False, ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Histogramme des consommations")
    fig, ax = plt.subplots()
    for segment, data, color in zip(['Low', 'Medium', 'High'],
                                    [low_df['conso_energie'], med_df['conso_energie'], high_df['conso_energie']],
                                    ['green', 'orange', 'red']):
        sns.histplot(data, kde=True, label=segment, color=color, stat='density', ax=ax)
    ax.legend()
    st.pyplot(fig)

    st.subheader("📈 Évolution temporelle")
    fig, ax = plt.subplots()
    low_df['conso_energie'].plot(ax=ax, label='Low', color='green')
    med_df['conso_energie'].plot(ax=ax, label='Medium', color='orange')
    high_df['conso_energie'].plot(ax=ax, label='High', color='red')
    ax.legend()
    st.pyplot(fig)

# --- Chargement des modèles
@st.cache_resource
def load_models():
    def train_knn(df_subset):
        scaler = StandardScaler()
        # 👉 Ajout de 'jour' dans les features pour entraînement
        X = scaler.fit_transform(df_subset[['temp_ext', 'ensoleillement', 'jour']])
        knn = NearestNeighbors(n_neighbors=min(5, len(df_subset)), metric='cosine', algorithm='brute')
        knn.fit(X)
        return knn, scaler, df_subset

    # Création des modèles KNN segmentés
    knn_low, sc_low, df_low = train_knn(low_df)
    knn_med, sc_med, df_med = train_knn(med_df)
    knn_high, sc_high, df_high = train_knn(high_df)

    # Modèle ExtraTrees
    X_train = df[['temp_ext', 'ensoleillement', 'jour', 'conso_energie']]
    y_train = df['temp_int']
    model_et = ExtraTreesRegressor(n_estimators=50, max_features='sqrt', random_state=42)
    model_et.fit(X_train, y_train)

    return {
        'knn': {
            'low': (knn_low, sc_low, df_low),
            'medium': (knn_med, sc_med, df_med),
            'high': (knn_high, sc_high, df_high)
        },
        'extra_trees': model_et
    }

models = load_models()

# --- Interface utilisateur enrichie
st.sidebar.header("⚙️ Paramètres prédiction")
mode = st.sidebar.selectbox("🎯 Mode de confort", ['Low', 'Medium', 'High'])
top_p = st.sidebar.slider("🔍 Niveau de similarité (top-p)", 0.01, 1.0, 0.3)

# 🗓️ Jour sélectionnable
jour = st.sidebar.selectbox(
    "📅 Jour de la semaine",
    options=range(7),
    format_func=lambda x: ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"][x]
)

# --- Données d'entrée pour prédiction KNN
input_features_df = pd.DataFrame([{
    'temp_ext': temp_ext,
    'ensoleillement': ensoleillement,
    'jour': jour
}])

# --- Prédiction avec KNN
knn_model, scaler, df_ref = models['knn'][mode.lower()]
scaled_input = scaler.transform(input_features_df)
n_neighbors = min(5, len(df_ref))
distances, indices = knn_model.kneighbors(scaled_input, n_neighbors=n_neighbors)
filtered_indices = [i for i, d in zip(indices[0], distances[0]) if d <= (1 - top_p)]

if not filtered_indices:
    st.warning("🔄 Aucun voisin trouvé, fallback activé.")
    filtered_indices = list(indices[0])

conso_estimee = df_ref.iloc[filtered_indices]['conso_energie'].mean()

# --- Affichage consommation estimée
st.subheader("🔌 Consommation énergétique estimée")
st.metric("Estimation (kWh)", f"{conso_estimee:.2f} kWh")

# --- Prédiction température recommandée
input_et = pd.DataFrame([{
    'temp_ext': temp_ext,
    'ensoleillement': ensoleillement,
    'jour': jour,
    'conso_energie': conso_estimee
}])
temp_pred = models['extra_trees'].predict(input_et)[0]

st.subheader("🌡️ Température intérieure recommandée")
st.metric("Consigne (°C)", f"{temp_pred:.2f} °C")

# --- Résultats
st.subheader("🔌 Consommation énergétique estimée")
st.metric("Estimation (kWh)", f"{conso_estimee:.2f} kWh")

input_df = pd.DataFrame([{
    'temp_ext': temp_ext,
    'ensoleillement': ensoleillement,
    'jour': jour,
    'conso_energie': conso_estimee
}])
temp_pred = models['extra_trees'].predict(input_df)[0]

st.subheader("🌡️ Température intérieure recommandée")
st.metric("Consigne (°C)", f"{temp_pred:.2f} °C")
# --- Visualisation complète
with st.expander("🔎 🧾 📈 Données par niveau de consommation"):
    st.write("🔋 Faible consommation")
    st.dataframe(low_df[['temp_ext', 'ensoleillement', 'conso_energie']])
    st.write("⚖️ Moyenne consommation")
    st.dataframe(med_df[['temp_ext', 'ensoleillement', 'conso_energie']])
    st.write("🔥 Forte consommation")
    st.dataframe(high_df[['temp_ext', 'ensoleillement', 'conso_energie']])

with st.expander("🔎 🧾 📈 Données de consommation et de température historique (Plotly)"):
    # --- Consommation - Boxplot
    st.subheader("📦 Boxplot consommation par segment")
    df_plot = pd.DataFrame({
        'Conso (kWh)': pd.concat([low_df['conso_energie'], med_df['conso_energie'], high_df['conso_energie']]),
        'Segment': ['Low'] * len(low_df) + ['Medium'] * len(med_df) + ['High'] * len(high_df)
    })
    fig_box_conso = px.box(df_plot, x='Segment', y='Conso (kWh)', color='Segment', color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_box_conso, use_container_width=True)

    # --- Consommation - Histogramme
    st.subheader("📊 Histogramme des consommations")
    fig_hist_conso = go.Figure()
    for segment, data, color in zip(
        ['Low', 'Medium', 'High'],
        [low_df['conso_energie'], med_df['conso_energie'], high_df['conso_energie']],
        ['green', 'orange', 'red']
    ):
        fig_hist_conso.add_trace(go.Histogram(
            x=data,
            name=segment,
            opacity=0.6,
            histnorm='probability density',
            marker_color=color
        ))
    fig_hist_conso.update_layout(barmode='overlay', xaxis_title="Consommation (kWh)", yaxis_title="Densité")
    st.plotly_chart(fig_hist_conso, use_container_width=True)

    # --- Consommation - Série temporelle
    st.subheader("📈 Évolution temporelle de la consommation")
    fig_ts_conso = go.Figure()
    fig_ts_conso.add_trace(go.Scatter(y=low_df['conso_energie'], name='Low', mode='lines', line=dict(color='green')))
    fig_ts_conso.add_trace(go.Scatter(y=med_df['conso_energie'], name='Medium', mode='lines', line=dict(color='orange')))
    fig_ts_conso.add_trace(go.Scatter(y=high_df['conso_energie'], name='High', mode='lines', line=dict(color='red')))
    fig_ts_conso.update_layout(yaxis_title="kWh")
    st.plotly_chart(fig_ts_conso, use_container_width=True)

    # --- Température extérieure - Boxplot
    st.subheader("📦 Boxplot température extérieure par segment")
    df_plot_temp = pd.DataFrame({
        'Température ext (°C)': pd.concat([low_df['temp_ext'], med_df['temp_ext'], high_df['temp_ext']]),
        'Segment': ['Low'] * len(low_df) + ['Medium'] * len(med_df) + ['High'] * len(high_df)
    })
    fig_box_temp = px.box(df_plot_temp, x='Segment', y='Température ext (°C)', color='Segment', color_discrete_sequence=px.colors.diverging.Tealrose)
    st.plotly_chart(fig_box_temp, use_container_width=True)

    # --- Température extérieure - Histogramme
    st.subheader("🌡️ Histogramme des températures extérieures")
    fig_hist_temp = go.Figure()
    for segment, data, color in zip(
        ['Low', 'Medium', 'High'],
        [low_df['temp_ext'], med_df['temp_ext'], high_df['temp_ext']],
        ['blue', 'purple', 'red']
    ):
        fig_hist_temp.add_trace(go.Histogram(
            x=data,
            name=segment,
            opacity=0.6,
            histnorm='probability density',
            marker_color=color
        ))
    fig_hist_temp.update_layout(barmode='overlay', xaxis_title="Température extérieure (°C)", yaxis_title="Densité")
    st.plotly_chart(fig_hist_temp, use_container_width=True)

    # --- Température extérieure - Série temporelle
    st.subheader("📈 Évolution temporelle de la température extérieure")
    fig_ts_temp = go.Figure()
    fig_ts_temp.add_trace(go.Scatter(y=low_df['temp_ext'], name='Low', mode='lines', line=dict(color='blue')))
    fig_ts_temp.add_trace(go.Scatter(y=med_df['temp_ext'], name='Medium', mode='lines', line=dict(color='purple')))
    fig_ts_temp.add_trace(go.Scatter(y=high_df['temp_ext'], name='High', mode='lines', line=dict(color='red')))
    fig_ts_temp.update_layout(yaxis_title="°C")
    st.plotly_chart(fig_ts_temp, use_container_width=True)

