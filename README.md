# 🎮 IndieGuide — Asistente experto en videojuegos indie

**Proyecto de IA Generativa · Máster en Data Science & IA (Evolve)**
Autor: Fernando López Sánchez ([@FLopez26](https://github.com/FLopez26))

IndieGuide es un agente conversacional que **recomienda videojuegos indie y responde preguntas sobre ellos y sobre sus estudios**. Para responder usa **solo** su propia base de conocimiento vectorial. Combina **Google Gemini** (LLM y embeddings), **ChromaDB** como base vectorial y **LangGraph** para orquestar el agente, con memoria de conversación. Incluye una interfaz web en **Streamlit**.

---

## 📑 Índice
1. [Dominio elegido](#-dominio-elegido)
2. [Arquitectura](#️-arquitectura)
3. [Stack tecnológico](#-stack-tecnológico)
4. [Estructura del repositorio](#-estructura-del-repositorio)
5. [Datos: origen y preparación](#-datos-origen-y-preparación)
6. [Instalación](#-instalación)
7. [Ejecución](#️-ejecución)
8. [System prompt y justificación](#-system-prompt-y-justificación)
9. [Decisiones técnicas](#-decisiones-técnicas)
10. [Preguntas de ejemplo](#-preguntas-de-ejemplo)
11. [Limitaciones y mejoras](#-limitaciones-y-mejoras)

---

## 🎯 Dominio elegido

**Videojuegos indie.** En Steam hay decenas de miles de juegos independientes y es difícil encontrar uno que encaje con lo que apetece jugar. IndieGuide actúa como un experto que:

- **Recomienda juegos** según género, ambientación, modo de juego (solo o cooperativo) o precio.
- **Responde preguntas** sobre un juego concreto: de qué trata, cuánto cuesta, cómo lo valoran los jugadores o cuánto tiempo suele jugarse.
- **Informa sobre estudios** indie: de dónde son y qué otros juegos han hecho.
- **Mantiene el contexto** de la conversación ("¿y alguno más barato?", "¿quién hizo el primero?").
- **No inventa:** si un juego no está en su catálogo, lo dice.

---

## 🏗️ Arquitectura

```
                         ┌──────────────────────────────┐
  Pregunta del usuario ─►│  LangGraph                   │
                         │                              │
                         │  START                       │
                         │    │                         │
                         │    ▼                         │
                         │  reformular ── Gemini LLM    │  "¿y alguno más barato?"
                         │    │                         │   → "roguelike rápido y barato"
                         │    ▼                         │
                         │  recuperar ─── ChromaDB ◄────┼── Gemini Embeddings
                         │    │          (top 6 docs)   │
                         │    ▼                         │
                         │  generar ───── Gemini LLM    │  system prompt + contexto
                         │    │                         │  + últimos 10 mensajes
                         │    ▼                         │
                         │   END                        │
                         │                              │
                         │  Checkpointer (InMemorySaver)│  memoria por thread_id
                         └──────────────────────────────┘
```

| Nodo | Función |
|---|---|
| **reformular** | Convierte la última pregunta en una consulta autónoma usando el historial. Sin este paso, preguntas como *"¿y alguno más barato?"* recuperarían documentos irrelevantes. |
| **recuperar** | Busca en ChromaDB los 6 documentos más similares a la consulta reformulada. Se ejecuta **en cada turno**, así que el agente siempre usa la base de conocimiento. |
| **generar** | Llama a Gemini con el system prompt, el contexto recuperado (entre etiquetas `<contexto>`) y los últimos 10 mensajes del historial. |

---

## 🧰 Stack tecnológico

| Componente | Tecnología | Uso |
|---|---|---|
| LLM | Google Gemini `gemini-3.5-flash-lite` | Reformulación de preguntas y generación de respuestas |
| Embeddings | Google Gemini `gemini-embedding-2` | Vectores de 3.072 dimensiones, multilingües |
| Base vectorial | ChromaDB (persistente) | Almacenamiento y búsqueda por similitud |
| Agente | LangGraph + LangChain | Grafo de nodos con estado compartido |
| Memoria | `InMemorySaver` (LangGraph) | Historial por conversación (`thread_id`) |
| Datos | pandas | Limpieza y filtrado del dataset |
| Interfaz | Streamlit | Chat web (bonus) |
| Entorno | Python 3.13 + `venv` + `python-dotenv` | Dependencias aisladas y API key segura |

---

## 📁 Estructura del repositorio

```
proyectoIA/
├── proyecto_indieguide.ipynb      # Notebook principal (entregable)
├── app.py                         # Interfaz web en Streamlit
├── requirements.txt               # Dependencias
├── README.md
├── .gitignore
├── .env                           # API key (NO se sube al repositorio)
└── data/
    ├── juegos_indie.csv           # 200 juegos indie filtrados (fuente principal)
    ├── indie-games-developers.csv # 200 estudios indie (fuente secundaria)
    └── chroma/                    # Base vectorial ya indexada (400 documentos)
```

---

## 📊 Datos: origen y preparación

La base de conocimiento combina **dos fuentes**, con **400 documentos** en total (~80 páginas de texto equivalente).

### 1. `juegos_indie.csv`: 200 juegos (fuente principal)

Se obtuvo a partir del [Steam Games Dataset (Kaggle)](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset), que tiene 125.855 juegos y 39 columnas. La preparación se hizo una sola vez y no forma parte del notebook:

1. **Corrección de la cabecera:** el CSV original tiene una coma de menos (`DiscountDLC count`). Por eso la cabecera tiene 39 columnas y las filas 40, y todos los datos quedan desplazados una posición. Se corrigió separando `Discount` y `DLC count`.
2. **Filtros:**
   - Género *Indie*.
   - Sin demos, playtests, bandas sonoras ni servidores dedicados.
   - Al menos 2.000 reseñas y al menos 85 % de reseñas positivas.
   - Descripción de al menos 300 caracteres.
3. **Selección:** los 200 juegos con más reseñas positivas.
4. **Columnas derivadas:**
   - Año de lanzamiento.
   - Mediana de horas jugadas (minutos → horas).
   - Los 10 tags principales.
   - Descripción limpia y recortada a 1.200 caracteres.
5. **Cruce con la fuente 2:** se añadieron ciudad y país del estudio. Hubo 27 coincidencias.

### 2. `indie-games-developers.csv`: 200 estudios (fuente secundaria)

Es un listado de estudios de desarrollo indie (procedente de Wikipedia) con ciudad, país, juegos destacados y notas. Se eliminaron las referencias tipo `[1]`. Permite al agente responder preguntas sobre estudios.

> **¿Por qué solo 200 juegos?** La capa gratuita de Gemini permite 100 embeddings por minuto. Un catálogo pequeño y de calidad (juegos populares y bien valorados) se indexa en unos 5 minutos y da respuestas más útiles que uno enorme con miles de juegos sin reseñas.

---

## 🔧 Instalación

### Requisitos
- **Python 3.13** (también funciona con 3.11 y 3.12).
- **API key de Google Gemini**, gratuita, desde [Google AI Studio](https://aistudio.google.com/app/apikey).
- **Git** y **VS Code** con la extensión de Jupyter (o Jupyter Notebook).

### Pasos

**1. Clonar el repositorio**
```bash
git clone https://github.com/FLopez26/indieguide.git
cd indieguide
```

**2. Crear y activar el entorno virtual**
```bash
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

**3. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**4. Configurar la API key**

Crea un archivo `.env` en la raíz del proyecto con esta línea:
```
GOOGLE_API_KEY=tu_clave_de_gemini
```
El archivo `.env` está en `.gitignore` y nunca se sube al repositorio.

---

## ▶️ Ejecución

### Notebook (entregable principal)

1. Abre `proyecto_indieguide.ipynb` en VS Code y selecciona el kernel `.venv`.
2. Ejecuta las celdas en orden. Las secciones son:

| Sección | Contenido |
|---|---|
| 1. Configuración | Carga de la API key, modelos y prueba de conexión |
| 2. Base de conocimiento | Construcción de los 400 documentos, indexación en ChromaDB y pruebas de recuperación |
| 3. System prompt | Prompt del agente y técnicas aplicadas |
| 4. Agente LangGraph | Estado, nodos, grafo, visualización y traza paso a paso |
| 5. Memoria | Conversación encadenada y aislamiento entre threads |
| 6. Ejemplos | 6 preguntas documentadas con su análisis |
| 7. Chat interactivo | Conversación libre (escribe `salir` para terminar) |
| 8. Conclusiones | Resultados y mejoras |

**Sobre la indexación:**
- El repositorio ya incluye `data/chroma` indexado, así que la celda de indexación detecta los 400 documentos y no gasta cuota.
- Para re-indexar desde cero, borra la carpeta `data/chroma`. Tarda unos 5 minutos por el límite de la API.

### Interfaz web (Streamlit)

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. La app reutiliza la base vectorial de `data/chroma`. Incluye:
- Chat con memoria por sesión.
- Preguntas de ejemplo en el panel lateral.
- Un desplegable **"Detalles del RAG"** con la consulta reformulada y los documentos recuperados en cada respuesta.
- Un botón para empezar una conversación nueva.

**Despliegue en Streamlit Cloud:**
1. Conecta el repositorio en [share.streamlit.io](https://share.streamlit.io).
2. Elige `app.py` y Python 3.13 en *Advanced settings*.
3. Añade en *Secrets*: `GOOGLE_API_KEY = "tu_clave"`

---

## 📝 System prompt y justificación

```text
Eres IndieGuide, un experto en videojuegos indie que ayuda a los usuarios a descubrir juegos que encajen con sus gustos.

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
```

### Justificación

| Decisión | Técnica | Motivo |
|---|---|---|
| Rol fijo "IndieGuide, experto en videojuegos indie" | Patrón persona / root prompt | Da un tono coherente y especializado durante toda la conversación |
| "Solo conoces lo que aparece en el CONTEXTO" + regla 1 | Verificación de condiciones | Es el núcleo del RAG: evita que Gemini use su conocimiento general e invente datos |
| Contexto entre `<contexto>` y `</contexto>` | Delimitadores | Separa claramente las instrucciones de los datos recuperados |
| Regla 2: "No lo tengo en mi catálogo" + alternativa | Verificación de condiciones | Define qué hacer cuando no hay información suficiente, sin cortar la conversación |
| Regla 3: justificar cada recomendación | Instrucciones explícitas | Una recomendación sin motivo no aporta valor; obliga a conectar los datos con lo que pide el usuario |
| Regla 4: máximo 3 juegos, ordenados | Meta lenguaje | Respuestas concisas y priorizadas, sin listas interminables |
| Regla 5: horas jugadas ≠ duración | Contexto del dominio | La mediana de horas de Steam se dispara en juegos multijugador o sandbox (Terraria: 34 h, Rust: 51 h). Sin esta aclaración el agente diría que un juego "dura" 50 horas |
| Regla 6: precios en USD y orientativos | Contexto del dominio | Los precios del dataset son una foto fija y pueden haber cambiado |
| Regla 7: peticiones vagas → recomendar + preguntar | Comportamiento conversacional | Da valor inmediato y a la vez afina los gustos del usuario |
| Regla 8: solo videojuegos | Filtro semántico | Mantiene al agente dentro de su dominio |
| Regla 9: tener en cuenta la conversación | Coherencia conversacional | Refuerza el uso de la memoria en la generación |
| Formato fijo por recomendación | Salida estructurada | Respuestas homogéneas y fáciles de leer |
| Traducir y resumir las descripciones | Meta lenguaje | El catálogo está en inglés y el usuario conversa en español |

---

## 🧠 Decisiones técnicas

| Decisión | Motivo |
|---|---|
| **Sin chunking** (1 documento = 1 juego o 1 estudio) | Cada juego ya es una unidad completa y acotada (≤ 1.200 caracteres de descripción). Partirlo separaría el nombre de su descripción y el retriever devolvería fragmentos sin saber de qué juego son. |
| **Texto estructurado en `page_content`** (`Juego:`, `Precio:`, `Tags:`…) | Facilita que el embedding capture los atributos clave y que el LLM los lea sin ambigüedad. |
| **Metadatos** (`tipo`, `precio`, `anio`, `pct_positivas`…) | Permiten filtrar en ChromaDB (por ejemplo `{"precio": {"$lt": 10}}`) y saber qué documentos se han usado. |
| **Indexación en lotes de 50 con 40 s de pausa** | La capa gratuita de Gemini permite 100 embeddings por minuto. Con más velocidad salta el error `429 RESOURCE_EXHAUSTED`. |
| **IDs fijos por documento** (`juego-0`, `estudio-15`…) | Si la indexación se interrumpe, al volver a ejecutarla se indexan solo los documentos que faltan, sin duplicados. |
| **ChromaDB persistente en disco** | Se indexa una vez y después solo se lee, sin coste de tiempo ni de cuota. |
| **Nodo `reformular`** | La memoria tiene que llegar también a la búsqueda, no solo a la generación. Buscar *"¿y alguno más barato?"* tal cual devolvería documentos sin relación. |
| **Historial recortado a 10 mensajes** (*trim*) | Evita que el contexto enviado al LLM crezca sin límite en conversaciones largas. |
| **`k = 6` documentos recuperados** | Margen suficiente para elegir 3 recomendaciones y para que entren documentos de estudios cuando son relevantes. |
| **Modelo `gemini-3.5-flash-lite`** | Los modelos 2.5 ya no están disponibles para cuentas nuevas. En la familia 3.x la temperatura es fija y la respuesta llega como lista de bloques, por eso se usa `.text` en lugar de `.content`. |

---

## 💬 Preguntas de ejemplo

Todas están documentadas en las secciones 5 y 6 del notebook.

| # | Pregunta | Qué demuestra |
|---|---|---|
| 1 | *"Quiero algo relajante para jugar por las noches, sin combate ni estrés"* | Recomendación por ambiente |
| 2 | *"Recomiéndame un juego cooperativo para jugar con un amigo"* | Recomendación por modo de juego |
| 3 | *"¿Qué sabes de Klei Entertainment? ¿Qué juegos ha hecho?"* | Uso de la fuente secundaria (estudios) |
| 4 | *"Háblame de Hollow Knight: ¿de qué va y cuánto suele jugar la gente?"* | Pregunta sobre un juego concreto y regla de horas jugadas |
| 5 | *"¿Qué tal está Elden Ring?"* | Juego fuera del catálogo: no inventa |
| 6 | *"¿Cuál es la capital de Australia?"* | Filtro semántico: pregunta fuera de tema |
| M1 | *"Busco un roguelike con combate rápido y que sea muy rejugable"* | Inicio de la conversación con memoria |
| M2 | *"¿Y alguno parecido que sea más barato?"* | Memoria y reformulación de la consulta |
| M3 | *"¿Qué estudio hizo el primero que me recomendaste al principio y de dónde es?"* | Memoria de varios turnos |

---

## 🚧 Limitaciones y mejoras

**Limitaciones**
- El catálogo es una foto fija: precios y valoraciones no se actualizan.
- Las descripciones están en inglés y el agente las traduce al responder.
- La memoria (`InMemorySaver`) se pierde al reiniciar el notebook o la app.
- La capa gratuita de Gemini limita el número de peticiones por minuto.

**Posibles mejoras**
- **Filtros automáticos:** extraer del mensaje restricciones como "menos de 10 €" o "después de 2020" y aplicarlas como filtro en ChromaDB.
- **Memoria persistente:** usar un checkpointer en SQLite para conservar las conversaciones entre sesiones.
- **Router de intención:** un nodo condicional que distinga recomendación, pregunta sobre un estudio y charla, y evite la búsqueda cuando no hace falta.
- **Catálogo más amplio:** más juegos y descripciones completas, con plan de pago o indexación progresiva.

---

## 📦 Dependencias

```
langchain
langchain-core
langchain-google-genai
langgraph
langchain-chroma
chromadb
langchain-community
langchain-text-splitters
pypdf
pandas
python-dotenv
ipykernel
streamlit
```

Las versiones exactas usadas en el desarrollo están en `requirements-lock.txt`.
