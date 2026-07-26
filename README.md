# 🏔️ Agente Turístico de Santander — Challenge Alura Agente

Agente de inteligencia artificial que responde preguntas sobre los **sitios turísticos
del departamento de Santander, Colombia**, a partir del contenido de un documento PDF.
Está orquestado con **LangChain** y puede usar varios modelos de lenguaje (**Gemini**,
**Groq** o **Cohere**) con respaldo automático entre ellos. Se expone mediante una **API con
FastAPI** con interfaz web de chat, incluye **análisis de imágenes** (Gemini visión) y un
**agente con herramientas**, y se despliega en **Oracle Cloud Infrastructure (OCI Compute)**.

> Proyecto desarrollado para el **Challenge Alura Agente**.

---

## 📖 Descripción general

El agente permite hacer preguntas en lenguaje natural (por ejemplo, *"¿dónde puedo practicar
rafting?"*) y obtener respuestas fundamentadas exclusivamente en una guía turística en formato
PDF, sin necesidad de abrir el documento.

**Estrategia: inyección de contexto completo.** Como el documento fuente es pequeño (una guía
de 5 páginas), el agente entrega el **texto completo del PDF como contexto** al modelo en cada
pregunta. Aprovechando la amplia ventana de contexto de los LLM actuales, esto resulta más
simple, preciso y económico que un pipeline de embeddings para un documento de este tamaño, y
evita que el modelo invente información fuera del documento.

---

## 🏗️ Arquitectura de la solución

```
                          ┌─────────────────────────────────────────┐
   Usuario  ──HTTP──────▶ │              FastAPI (app)               │
 (navegador / API)        │  /  /ask  /vision  /history  /admin  /agente │
                          └───────┬───────────────┬─────────────────┘
                                  │               │
              ┌───────────────────┘               └───────────────┐
              ▼                                                    ▼
   ┌────────────────────┐   PDF (contexto)        ┌───────────────────────────┐
   │  Agente Q&A        │◀────── PDF Loader        │   Memoria (CSV + pandas)  │
   │  (contexto completo)│       (pypdf)           │   filtro por session_id   │
   └─────────┬──────────┘                          └───────────────────────────┘
             │  prompt (documento + memoria + pregunta)
             ▼
   ┌──────────────────────────────────────────────────────────────┐
   │   Capa LLM (LangChain, LCEL)  — estrategia de respaldo         │
   │   Groq (Llama/Gemma)  →  Gemini  →  Cohere    (según cupo)     │
   └──────────────────────────────────────────────────────────────┘
             │
             ▼
     Respuesta en lenguaje natural

  Extras:  /vision → Gemini multimodal (JSON estructurado)
           /agente → agente ReAct (LangGraph) que elige herramientas
```

**Flujo:**
1. **Al iniciar:** se lee el PDF y su texto queda cargado en memoria como contexto.
2. **Por cada pregunta:** se arma el prompt (documento + memoria + pregunta) y se envía a la
   capa LLM (LangChain), que elige el proveedor y aplica el respaldo si uno se queda sin cupo.
3. **Memoria por sesión (CSV + pandas):** cada usuario se identifica con un `session_id`
   generado en el frontend (sin registro). Cada intercambio se guarda como una fila en
   `data/historial.csv`. **Antes de responder**, el sistema lee el CSV y **filtra por
   `session_id`** (`df[df["session_id"] == sid]`) para recuperar la conversación previa y
   dar continuidad. El endpoint `GET /history` permite listar sesiones, ver una sesión o
   **buscar** un término (`?q=...`, filtro `str.contains`).
4. **Análisis de conversaciones (admin):** una acción protegida con clave
   (`POST /admin/analisis`, botón en el frontend) lee el historial con pandas, pide al LLM
   que **clasifique las preguntas en categorías devolviendo JSON**, lo convierte con
   `json.loads` y responde qué temas consultan más los usuarios. Aplica pandas + LLM + JSON.
5. **Análisis de imágenes (visión):** el usuario sube una foto (`POST /vision`, botón en el
   frontend); se codifica en base64 y se envía a **Gemini visión** (mensaje multimodal con
   LangChain) para identificar lugares, platos o actividades de Santander. Usa la mejor
   práctica de salida estructurada, `modelo.with_structured_output(ModeloPydantic)` (con un
   campo `Literal` para el tipo), devolviendo un objeto validado
   (`titulo`, `descripcion`, `etiquetas`, `tipo`, `relacion_santander`). Solo con Gemini.

### Estructura del repositorio

```
alura-latam/
├── app/
│   ├── main.py          # API FastAPI (endpoints + interfaz)
│   ├── agent.py         # Lógica del agente Q&A (carga del PDF + llamada al LLM)
│   ├── pdf_loader.py    # Lectura y limpieza del texto del PDF
│   ├── memory.py        # Memoria de conversaciones en CSV (pandas + filtros)
│   ├── analytics.py     # Análisis de conversaciones (pandas + LLM + JSON)
│   ├── vision.py        # Análisis de imágenes con Gemini visión (multimodal)
│   ├── rag.py           # RAG real (chunks + embeddings + FAISS/Pinecone) — /rag/ask
│   ├── llm.py           # Capa LLM con LangChain (Gemini/Groq/Cohere + fallback)
│   ├── tools.py         # Herramientas del agente orquestador (LangChain Tools)
│   ├── orchestrator.py  # Agente ReAct con LangGraph (endpoint /agente, paralelo)
│   ├── graph.py         # Agente con grafo de estados: triaje + RAG (/grafo/ask)
│   └── config.py        # Configuración desde variables de entorno
├── static/index.html    # Interfaz web de chat
├── tests/test_agent.py  # Tests unitarios (sin llamar a la API)
├── data/
│   ├── guia_turistica_santander.pdf   # Documento fuente
│   ├── guia_turistica_santander.md    # Versión editable
│   └── historial.csv                  # Memoria de conversaciones (generado en runtime)
├── docs/                # Diagramas y capturas del deploy
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## 🛠️ Tecnologías y herramientas

| Categoría        | Herramienta                                   |
|------------------|-----------------------------------------------|
| Lenguaje         | Python 3.11+                                  |
| API web          | FastAPI + Uvicorn                             |
| Orquestación LLM | **LangChain** (chat models + prompts + LCEL + fallback) |
| IA / LLM         | **Gemini**, **Groq** (Llama/Gemma) o **Cohere** — conmutable, con respaldo |
| Lectura de PDF   | pypdf                                         |
| RAG / vectores   | **FAISS** (persistido) o **Pinecone** (nube) + embeddings de Cohere |
| Memoria / datos  | pandas (CSV de sesiones, filtros)             |
| Nube / Deploy    | Oracle Cloud Infrastructure (OCI Compute)     |
| Frontend         | HTML + CSS + JavaScript (vanilla)             |
| Testing          | pytest                                        |

---

## 🚀 Instrucciones para ejecutar el proyecto

### 1. Requisitos previos
- Python 3.11 o superior.
- Una **API key gratuita** de al menos un proveedor de LLM (puedes usar los tres y el sistema
  alterna entre ellos):
  - **Google Gemini** — [Google AI Studio](https://aistudio.google.com/app/apikey) (requerido para la visión).
  - **Groq** — [console.groq.com/keys](https://console.groq.com/keys) (rápido, free tier generoso).
  - **Cohere** — [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) (fuerte en español).

### 2. Clonar e instalar dependencias
```bash
git clone https://github.com/AndrewCarvajal97/agente-turistico-santander.git
cd agente-turistico-santander
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar la variable de entorno
```bash
cp .env.example .env
```
Edita `.env` y coloca tu clave según el proveedor que elijas:

**Opción A — Google Gemini** (por defecto):
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...tu_clave...
```

**Opción B — Groq** (modelos open source, free tier más generoso y muy rápido):
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...tu_clave...
```
Consigue la clave de Groq (gratis, sin tarjeta) en [console.groq.com/keys](https://console.groq.com/keys).

### 4. Levantar la aplicación
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Abre **http://localhost:8000** para usar el chat, o **http://localhost:8000/docs** para la
documentación interactiva de la API.

### 5. (Opcional) Ejecutar los tests
```bash
python -m pytest -q
```

---

## ☁️ Despliegue en OCI

El agente se despliega en una **instancia Compute** de OCI (una VM). Los modelos de lenguaje
se consumen por API, por lo que basta con definir la clave del proveedor elegido (p. ej.
`GEMINI_API_KEY`, `GROQ_API_KEY` o `COHERE_API_KEY`) en el `.env` del servidor. Pasos resumidos:

1. Crear la VM (Oracle Linux / Ubuntu) y abrir el puerto `8000` en la *Security List* / *NSG*.
2. Conectarse por SSH y preparar el entorno:
   ```bash
   git clone https://github.com/AndrewCarvajal97/agente-turistico-santander.git
   cd agente-turistico-santander
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env          # edita .env y coloca la API key de tu proveedor
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. (Opcional) Contenerizar con el `Dockerfile` incluido y correr con Docker.

> 🔗 **Aplicación desplegada (OCI Compute — Bogotá):** http://149.130.177.89:8000
> 🖼️ **Captura del deploy:** ver `docs/captura-deploy.png`.

**Detalles del despliegue:**
- Región: **Colombia Central (Bogotá)** — `sa-bogota-1`
- Compartimento: `reto-alura` &nbsp;·&nbsp; VCN: `VCN-reto-alura` (subred pública)
- Instancia: `VM.Standard.E5.Flex` (Oracle Linux 9)
- Servidor: `uvicorn` en el puerto `8000` (abierto en firewalld + Security List)

---

## 💬 Ejemplos de preguntas que el agente puede responder

- ¿Cuál es la capital de Santander?
- ¿Cuál es el principal atractivo natural del departamento?
- ¿Dónde puedo practicar rafting y en qué ríos?
- ¿Por qué Barichara es tan famosa? ¿Qué es el Camino Real?
- ¿Qué comida típica debo probar en Santander?
- ¿Cuánto se tarda de Bucaramanga a San Gil?
- ¿Cuál es la mejor época para visitar?

## 🗨️ Ejemplos de respuestas generadas

> **Pregunta:** ¿Cuál es la capital de Santander?
>
> **Respuesta:** La capital de Santander es **Bucaramanga**, conocida como "La Ciudad Bonita"
> y "La Ciudad de los Parques".

> **Pregunta:** ¿Dónde puedo practicar rafting?
>
> **Respuesta:** Puedes practicar rafting en **San Gil**. El **río Fonce** es ideal para
> principiantes (nivel II-III) y el **río Suárez** para personas con experiencia (nivel IV-V).

> **Pregunta:** ¿Por qué Barichara es tan famosa?
>
> **Respuesta:** Barichara es considerada *"el pueblo más lindo de Colombia"* y está declarada
> Monumento Nacional. Destaca por su arquitectura colonial: calles empedradas y casas blancas
> de tapia pisada.

_(Las respuestas se generan dinámicamente con el LLM configurado, a partir del PDF fuente.)_

---

## 📄 Documento fuente

El agente responde a partir de [`data/guia_turistica_santander.pdf`](data/guia_turistica_santander.pdf),
una guía turística de 5 páginas que cubre destinos, deportes de aventura, gastronomía,
transporte, mejor época para visitar y preguntas frecuentes de Santander, Colombia.

---

## 🤖 Agente orquestador (ReAct) — implementación paralela

Además del ruteo explícito (`/ask`, `/vision`), el proyecto incluye un **agente ReAct**
opcional en `POST /agente`, construido con **LangGraph** (`create_react_agent`). En lugar de
un ruteo fijo, el agente **razona y decide** qué herramienta usar según la consulta. Las
herramientas ([tools.py](app/tools.py)) son `guia_turistica` (Q&A sobre el PDF),
`buscar_historial` (búsqueda en la memoria) y `explicar` (explicación didáctica de un tema,
usando **Cohere**). La lista es fácil de extender (p. ej. una futura herramienta de base de
datos), y cada herramienta puede usar el LLM que mejor le sirva.

> Es una vía **paralela** que no altera los endpoints principales. Consume más tokens (razona
> + actúa en varios pasos), por lo que está pensada para demostración y crecimiento futuro.

## 🔎 RAG real (embeddings + base vectorial) — vía paralela

Además del enfoque de **contexto completo** de `/ask`, el proyecto incluye un **RAG clásico**
en `POST /rag/ask` ([rag.py](app/rag.py)): se cargan **todos los PDFs** de una carpeta con
`DirectoryLoader` (multi-documento, escalable), se dividen en *chunks* (configurable:
`RecursiveCharacterTextSplitter` por caracteres, o **`SemanticChunker`** por significado),
cada chunk se convierte en un **vector semántico** con
**embeddings de Cohere** (o Gemini, configurable), y se indexa en una **base vectorial**. La recuperación
usa un *retriever* con **umbral de similitud** (`similarity_score_threshold`): por cada
pregunta se traen solo los chunks realmente relevantes y esos se pasan al LLM (con una cadena
*stuff* en LCEL: `prompt | modelo | StrOutputParser`) para **generar** la respuesta. La
respuesta es estructurada: `{respuesta, citaciones, documentos_encontrados}`. Cada **citación**
es trazable (fragmento + **página** + archivo, vía `PyPDFLoader` + `split_documents`). Si nada supera
el umbral —o si el modelo no halla la respuesta en el contexto— devuelve **"No lo sé"** con
`documentos_encontrados: false` (evita alucinar). Es la técnica adecuada para escalar a
documentos grandes o múltiples fuentes (requiere `COHERE_API_KEY`).

### 🗄️ Base vectorial intercambiable (patrón *strategy*)

El backend de la base vectorial se elige con `RAG_VECTORSTORE`, sin tocar el resto del pipeline:

| Backend | `RAG_VECTORSTORE` | Descripción |
|---|---|---|
| **FAISS** (por defecto) | `faiss` | Local y **persistido en disco** (`data/faiss_index/`): si el índice existe se **carga** en vez de recalcular los embeddings, así no se reindexa —ni se gasta cuota— en cada arranque. |
| **Pinecone** | `pinecone` | Base vectorial **en la nube** (serverless): los vectores persisten fuera del servidor y escalan a muchos documentos. Crea el índice automáticamente con la dimensión correcta (Cohere `embed-multilingual-v3.0` = 1024). Requiere `PINECONE_API_KEY`. |

Al **agregar o quitar PDFs**, `POST /rag/reindex` (protegido con `ADMIN_KEY`) reconstruye el
índice (`forzar=True`) para que tome los documentos actuales.

### 🔁 Multi-query (RAG avanzado, opcional)

Con `RAG_MULTIQUERY=true`, una LLM **reescribe la pregunta en varias versiones** (`RAG_MULTIQUERY_N`)
antes de buscar; se recuperan los chunks de **todas** las variantes y se **unen sin duplicados**.
Esto mejora el *recall* y supera las limitaciones de la búsqueda por distancia (una pregunta mal
delimitada encuentra igual el contexto). Es **opt-in** (suma una llamada al LLM por consulta) y
**degrada con elegancia**: si la generación de variantes falla, usa la pregunta original. Mantiene
intactas las citaciones y el chequeo "No lo sé".

### 🔭 Observabilidad con LangSmith (opcional)

El proyecto es compatible con **LangSmith** para *tracing*: ver cada paso de las cadenas
(chunks recuperados, prompt enviado, tokens, latencia). Es **opt-in** vía variables de entorno
(`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`); LangChain lo activa solo,
sin cambios de código, y sin ellas todo funciona igual. `GET /health` reporta si está activo.

## 🕸️ Agente con grafo de estados (LangGraph) — vía paralela

En `POST /grafo/ask` ([graph.py](app/graph.py)) hay un agente modelado como un **grafo de
estados** con `StateGraph` (LangGraph). El flujo es determinista y conecta todo lo construido:

```mermaid
graph TD
    START([inicio]) --> triaje
    triaje -. rag .-> auto_resolver
    triaje -. info .-> pedir_info
    triaje -. ticket .-> abrir_ticket
    auto_resolver -. ok .-> FIN([fin])
    auto_resolver -. info .-> pedir_info
    auto_resolver -. ticket .-> abrir_ticket
    pedir_info --> FIN
    abrir_ticket --> FIN
```

El **triaje** clasifica la consulta con `with_structured_output` (Pydantic + `Literal`:
`auto_resolver` / `pedir_info` / `abrir_ticket`) y una **arista condicional** enruta al nodo
correspondiente. Tras el **RAG** hay una **segunda arista condicional**: si respondió, termina;
si no, según la consulta abre un ticket (palabras como "reservar") o pide más info. El nodo de
ticket incluye la **urgencia** del triaje. El estado se propaga con un `TypedDict` (`AgentState`).
El endpoint `GET /grafo/diagrama` devuelve el grafo en Mermaid. Validado en vivo.

## 🗺️ Roadmap / próximos pasos

- Sumar herramientas al orquestador (p. ej. consulta a una base de datos).

---

## 📝 Licencia

Proyecto educativo desarrollado para el Challenge Alura Agente.
