"""
IndieGuide — Interfaz web en Streamlit
Ejecutar con:  streamlit run app.py
Requiere que la base vectorial (data/chroma) ya exista: se crea ejecutando el notebook.
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver


# --------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------
MODEL = "gemini-3.5-flash-lite"
MODEL_EMBEDDING = "models/gemini-embedding-2"
PERSIST_DIR = "data/chroma"
COLECCION = "juegos_indie"
K_DOCUMENTOS = 6
MAX_HISTORIAL = 10

PREGUNTAS_EJEMPLO = [
    "Busco un roguelike con combate rápido",
    "Algo relajante para jugar por las noches",
    "Un juego cooperativo para jugar con un amigo",
    "¿Qué juegos ha hecho Klei Entertainment?",
]

SYSTEM_PROMPT = """Eres IndieGuide, un experto en videojuegos indie que ayuda a los usuarios a descubrir juegos que encajen con sus gustos.

## Tu conocimiento
Solo conoces los juegos y estudios que aparecen en el CONTEXTO que se te proporciona en cada turno. Este contexto procede de un catálogo de 200 juegos indie bien valorados en Steam y de un listado de estudios indie.

## Reglas
1. Recomienda ÚNICAMENTE juegos que aparezcan en el contexto. Nunca inventes juegos, precios, valoraciones ni características.
2. Si el usuario pregunta por un juego o estudio que no está en el contexto, dilo claramente ("No lo tengo en mi catálogo") y, si es posible, ofrece una alternativa parecida del contexto.
3. Justifica cada recomendación relacionándola con lo que el usuario ha pedido (género, ambientación, precio, modo de juego…).
4. Recomienda como máximo 3 juegos por respuesta, ordenados de más a menos adecuado.
5. La "mediana de horas jugadas" indica cuánto suele jugar la gente, NO lo que dura completar el juego. Si la mencionas, preséntala como "tiempo de juego habitual". En juegos multijugador o sandbox este valor puede ser muy alto.
6. Los precios están en dólares (USD) y pueden haber cambiado; indícalo si el usuario pregunta específicamente por el precio.
7. Si la petición es muy vaga (por ejemplo "recomiéndame algo"), recomienda igualmente y termina con una pregunta corta para afinar sus gustos.
8. Si la pregunta no tiene relación con videojuegos, indica amablemente que solo puedes ayudar con videojuegos indie.
9. Ten en cuenta la conversación anterior: si el usuario se refiere a "ese juego", "el segundo" o "alguno más barato", usa lo que ya habéis hablado.

## Formato de respuesta
- Responde siempre en español, con un tono cercano y entusiasta pero conciso.
- Para cada recomendación usa este formato:
  **Nombre del juego** (año) — precio · valoración
  Una o dos frases explicando por qué encaja con lo que busca.
- Las descripciones del catálogo están en inglés: tradúcelas y resúmelas, no las copies.
"""

st.set_page_config(
    page_title="IndieGuide",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stat-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: bold; color: #4285f4; }
    .stat-label  { color: #8b8fa8; font-size: 0.85rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------
# API KEY
# --------------------------------------------
def obtener_api_key():
    """En Streamlit Cloud se lee de st.secrets; en local, del archivo .env"""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        load_dotenv()
        return os.getenv("GOOGLE_API_KEY")


# --------------------------------------------
# AGENTE (se construye una sola vez y se reutiliza)
# --------------------------------------------
class EstadoAgente(MessagesState):
    consulta: str
    contexto: str
    fuentes: list[str]


@st.cache_resource
def crear_agente(api_key: str):
    llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING, google_api_key=api_key)
    vectorstore = Chroma(
        collection_name=COLECCION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    def nodo_reformular(estado: EstadoAgente) -> dict:
        """Convierte la última pregunta en una consulta autónoma usando el historial"""
        mensajes = estado["messages"]
        pregunta = mensajes[-1].text
        if len(mensajes) == 1:
            return {"consulta": pregunta}

        historial = "\n".join(
            f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {m.text[:500]}"
            for m in mensajes[-7:-1]
        )
        prompt = f"""Tu tarea es reescribir la última pregunta del usuario como una consulta de búsqueda autónoma,
que se entienda sin leer la conversación. Sustituye referencias como "ese juego", "el segundo" o "alguno más barato"
por los nombres o características concretas mencionadas en el historial.

Responde SOLO con la consulta reescrita, sin explicaciones.

--- HISTORIAL ---
{historial}
--- FIN HISTORIAL ---

Última pregunta: {pregunta}
Consulta reescrita:"""
        return {"consulta": llm.invoke(prompt).text.strip()}

    def nodo_recuperar(estado: EstadoAgente) -> dict:
        """Recupera de ChromaDB los documentos más relevantes"""
        docs = vectorstore.similarity_search(estado["consulta"], k=K_DOCUMENTOS)
        contexto = "\n\n---\n\n".join(
            f"[Documento {i} - {d.metadata['tipo']}]\n{d.page_content}"
            for i, d in enumerate(docs, start=1)
        )
        fuentes = [f"{d.metadata['tipo']}: {d.metadata['nombre']}" for d in docs]
        return {"contexto": contexto, "fuentes": fuentes}

    def nodo_generar(estado: EstadoAgente) -> dict:
        """Genera la respuesta con system prompt + contexto + historial recortado"""
        system = SystemMessage(content=f"{SYSTEM_PROMPT}\n\n<contexto>\n{estado['contexto']}\n</contexto>")
        respuesta = llm.invoke([system] + estado["messages"][-MAX_HISTORIAL:])
        return {"messages": [AIMessage(content=respuesta.text)]}

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("reformular", nodo_reformular)
    grafo.add_node("recuperar", nodo_recuperar)
    grafo.add_node("generar", nodo_generar)
    grafo.add_edge(START, "reformular")
    grafo.add_edge("reformular", "recuperar")
    grafo.add_edge("recuperar", "generar")
    grafo.add_edge("generar", END)

    agente = grafo.compile(checkpointer=InMemorySaver())
    return agente, vectorstore._collection.count()


# --------------------------------------------
# ESTADO DE LA SESIÓN
# --------------------------------------------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []      # historial para pintar en pantalla
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"   # memoria del agente
    if "total_calls" not in st.session_state:
        st.session_state.total_calls = 0


def nueva_conversacion():
    st.session_state.messages = []
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.total_calls = 0


init_state()

api_key = obtener_api_key()
if not api_key:
    st.error("Falta la GOOGLE_API_KEY (archivo .env en local o Secrets en Streamlit Cloud).")
    st.stop()

agente, n_documentos = crear_agente(api_key)
if n_documentos == 0:
    st.error("La base vectorial está vacía. Ejecuta primero el notebook para indexar los documentos en data/chroma.")
    st.stop()


# --------------------------------------------
# SIDEBAR
# --------------------------------------------
with st.sidebar:
    st.title("🎮 IndieGuide")
    st.caption("Recomendador de videojuegos indie con Gemini, RAG y LangGraph")
    st.divider()

    st.subheader("💡 Prueba con")
    for q in PREGUNTAS_EJEMPLO:
        st.button(q, use_container_width=True,
                  on_click=lambda q=q: st.session_state.update(pregunta_ejemplo=q))

    st.divider()
    mostrar_detalle = st.toggle("Mostrar detalles del RAG", value=True,
                                help="Consulta reformulada y documentos recuperados en cada respuesta")

    st.subheader("📊 Sesión")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{st.session_state.total_calls}</div>"
                    f"<div class='stat-label'>Preguntas</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{n_documentos}</div>"
                    f"<div class='stat-label'>Documentos</div></div>", unsafe_allow_html=True)

    st.divider()
    st.button("🗑️ Nueva conversación", use_container_width=True, on_click=nueva_conversacion)


# --------------------------------------------
# MAIN
# --------------------------------------------
st.title("🎮 IndieGuide")
st.caption("Cuéntame qué te apetece jugar y te recomiendo juegos indie de mi catálogo de 200 títulos de Steam.")

if not st.session_state.messages:
    st.info("¡Empieza la conversación! Recordaré lo que hablemos en esta sesión.")


def pintar_detalle(msg):
    """Muestra la consulta reformulada y las fuentes recuperadas"""
    if mostrar_detalle and msg.get("fuentes"):
        with st.expander("🔍 Detalles del RAG"):
            st.markdown(f"**Consulta de búsqueda:** {msg['consulta']}")
            st.markdown("**Documentos recuperados:**")
            for f in msg["fuentes"]:
                st.markdown(f"- {f}")


# Historial de la conversación
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        pintar_detalle(msg)

# Entrada: chat o botón de ejemplo del sidebar
prompt = st.chat_input("¿Qué te apetece jugar?")
if not prompt:
    prompt = st.session_state.pop("pregunta_ejemplo", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el catálogo..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                resultado = agente.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

                msg = {
                    "role": "assistant",
                    "content": resultado["messages"][-1].text,
                    "consulta": resultado["consulta"],
                    "fuentes": resultado["fuentes"],
                }
                st.markdown(msg["content"])
                pintar_detalle(msg)

                st.session_state.messages.append(msg)
                st.session_state.total_calls += 1

            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    st.warning("⏳ Se ha alcanzado el límite de peticiones de la API gratuita. Espera un minuto y vuelve a intentarlo.")
                else:
                    st.error(f"❌ Error al llamar al agente: {e}")