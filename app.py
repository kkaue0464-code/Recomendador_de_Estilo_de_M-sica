import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
 
# ----------------------------------------------------------------------
# Configuração da página
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Mood Playlist Recommender",
    page_icon="🎧",
    layout="centered",
)
 
GENRES = ["Rock", "Hip Hop", "Rap", "Jazz"]
 
# ----------------------------------------------------------------------
# Catálogo de faixas sugeridas por gênero (exibição, não reprodução)
# ----------------------------------------------------------------------
CATALOG = {
    "Rock": [
        ("Foo Fighters", "Everlong"),
        ("Queen", "Don't Stop Me Now"),
        ("Arctic Monkeys", "Do I Wanna Know?"),
        ("Red Hot Chili Peppers", "Can't Stop"),
    ],
    "Hip Hop": [
        ("A Tribe Called Quest", "Can I Kick It?"),
        ("Anderson .Paak", "Come Down"),
        ("Kendrick Lamar", "Alright"),
        ("Chance the Rapper", "No Problem"),
    ],
    "Rap": [
        ("Eminem", "Till I Collapse"),
        ("Kendrick Lamar", "DNA."),
        ("J. Cole", "MIDDLE CHILD"),
        ("Nas", "N.Y. State of Mind"),
    ],
    "Jazz": [
        ("Miles Davis", "So What"),
        ("Bill Evans", "Waltz for Debby"),
        ("Nina Simone", "Feeling Good"),
        ("John Coltrane", "In a Sentimental Mood"),
    ],
}
 
GENRE_DESCRIPTIONS = {
    "Rock": "Energia alta e pouca tristeza — hora de faixas intensas e guitarras em evidência.",
    "Hip Hop": "Energia baixa e pouca tristeza — batidas envolventes para um clima leve e descontraído.",
    "Rap": "Energia alta e tristeza elevada — letras diretas e batidas fortes para colocar tudo pra fora.",
    "Jazz": "Energia baixa e tristeza elevada — sonoridade introspectiva e suave para desacelerar.",
}
 
 
# ----------------------------------------------------------------------
# Geração de dados sintéticos + treinamento do modelo (cacheado)
# ----------------------------------------------------------------------
def _regra_humor_para_genero(energia: float, tristeza: float) -> int:
    """Regra de negócio usada para gerar os rótulos de treino sintéticos."""
    if energia >= 5 and tristeza < 5:
        return GENRES.index("Rock")
    if energia < 5 and tristeza < 5:
        return GENRES.index("Hip Hop")
    if energia >= 5 and tristeza >= 5:
        return GENRES.index("Rap")
    return GENRES.index("Jazz")
 
 
@st.cache_resource(show_spinner=False)
def treinar_modelo() -> tf.keras.Model:
    """Treina um classificador denso simples (TensorFlow/Keras) sobre dados
    sintéticos derivados da regra de negócio, para mapear (energia, tristeza)
    -> gênero musical recomendado."""
    rng = np.random.default_rng(42)
    n_amostras = 4000
 
    energia = rng.uniform(0, 10, n_amostras)
    tristeza = rng.uniform(0, 10, n_amostras)
    # pequeno ruído para o modelo aprender uma fronteira suave, não apenas decorar
    ruido = rng.normal(0, 0.4, size=(n_amostras, 2))
 
    X = np.stack([energia, tristeza], axis=1) + ruido
    X = np.clip(X, 0, 10)
    y = np.array([_regra_humor_para_genero(e, t) for e, t in zip(energia, tristeza)])
 
    X_norm = X / 10.0  # normaliza para [0, 1]
 
    modelo = models.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(len(GENRES), activation="softmax"),
    ])
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.fit(X_norm, y, epochs=15, batch_size=32, verbose=0)
    return modelo
 
 
def prever_genero(modelo: tf.keras.Model, energia: float, tristeza: float):
    entrada = np.array([[energia / 10.0, tristeza / 10.0]], dtype=np.float32)
    probs = modelo.predict(entrada, verbose=0)[0]
    idx = int(np.argmax(probs))
    return GENRES[idx], dict(zip(GENRES, probs.tolist()))
 
 
# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
st.title("🎧 Mood Playlist Recommender")
st.caption("Ferramenta interna · Recomendação de gênero musical baseada no humor do momento")
 
st.markdown(
    "Ajuste os controles abaixo para refletir como você está se sentindo agora. "
    "O modelo vai recomendar o gênero musical e uma seleção de faixas alinhadas ao seu humor."
)
 
col1, col2 = st.columns(2)
with col1:
    energia = st.slider("Energia", min_value=0, max_value=10, value=5,
                         help="0 = sem disposição / 10 = extremamente energético")
with col2:
    tristeza = st.slider("Tristeza", min_value=0, max_value=10, value=5,
                          help="0 = ótimo humor / 10 = bastante triste")
 
gerar = st.button("Gerar recomendação", type="primary", use_container_width=True)
 
st.divider()
 
if gerar or "ultimo_genero" in st.session_state:
    modelo = treinar_modelo()
    genero, probs = prever_genero(modelo, energia, tristeza)
    st.session_state["ultimo_genero"] = genero
 
    st.subheader(f"Recomendação: {genero}")
    st.write(GENRE_DESCRIPTIONS[genero])
 
    with st.expander("Ver confiança do modelo por gênero"):
        df_probs = pd.DataFrame(
            {"Gênero": list(probs.keys()), "Confiança": [round(v * 100, 1) for v in probs.values()]}
        ).sort_values("Confiança", ascending=False).reset_index(drop=True)
        st.dataframe(df_probs, use_container_width=True, hide_index=True)
 
    st.markdown("### Faixas sugeridas")
    for artista, faixa in CATALOG[genero]:
        st.markdown(f"- **{faixa}** — {artista}")
else:
    st.info("Ajuste os sliders e clique em **Gerar recomendação** para ver sua playlist sugerida.")
 
st.divider()
st.caption("Modelo de classificação treinado localmente a cada sessão com TensorFlow/Keras · Uso interno")