import streamlit as st
import pandas as pd
from groq import Groq

st.set_page_config(page_title="Dashboard ENA 2024", layout="wide")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_data
def cargar_datos():
    df = pd.read_csv("03_CAP200A.csv", encoding="utf-8")
    df = df[["NOMBREDD", "P204_NOM", "P218C_PROD_CORTE_ENT"]].copy()
    df.columns = ["departamento", "cultivo", "produccion"]
    df["produccion"] = pd.to_numeric(df["produccion"], errors="coerce").fillna(0)
    return df

df = cargar_datos()

st.title("🌾 Dashboard Inteligente Agropecuario - ENA 2024")
st.markdown("Sistema de visualización con Big Data e IA para el sector agropecuario")

st.header("📊 Visualización de datos")
col1, col2, col3 = st.columns(3)
col1.metric("Total registros", f"{len(df):,}")
col2.metric("Departamentos", df["departamento"].nunique())
col3.metric("Producción total", f"{df['produccion'].sum():,.0f}")

st.subheader("Producción total por departamento")
prod_depto = df.groupby("departamento")["produccion"].sum().sort_values(ascending=False)
st.bar_chart(prod_depto)

st.subheader("Top 10 cultivos por producción")
top_cultivos = df.groupby("cultivo")["produccion"].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_cultivos)

st.header("🤖 Asistente IA Agropecuario")
st.markdown("Pregunta lo que quieras sobre los datos de la ENA 2024")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["texto"])

pregunta = st.chat_input("Escribe tu pregunta...")
if pregunta:
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    resumen = prod_depto.head(10).to_string()
    contexto = f"""Eres un asistente experto en el sector agropecuario peruano.
    Responde basándote en estos datos de producción por departamento (ENA 2024):
    {resumen}
    Pregunta del usuario: {pregunta}"""

    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": contexto}]
    )
    texto = respuesta.choices[0].message.content
    with st.chat_message("assistant"):
        st.markdown(texto)
    st.session_state.mensajes.append({"rol": "assistant", "texto": texto})
