# 🏔️ Agente Turístico de Santander — Challenge Alura Agente

Agente de inteligencia artificial que responde preguntas sobre los **sitios turísticos
del departamento de Santander, Colombia**, a partir del contenido de un documento PDF.
Usa **Google Gemini** como modelo de lenguaje, está expuesto mediante una **API con FastAPI**
con una interfaz web de chat, y se despliega en **Oracle Cloud Infrastructure (OCI Compute)**.

> Proyecto desarrollado para el **Challenge Alura Agente**.

---

## 📖 Descripción general

El agente permite hacer preguntas en lenguaje natural (por ejemplo, *"¿dónde puedo practicar
rafting?"*) y obtener respuestas fundamentadas exclusivamente en una guía turística en formato
PDF, sin necesidad de abrir el documento.

**Estrategia: inyección de contexto completo.** Como el documento fuente es pequeño (una guía
de 5 páginas), el agente entrega el **texto completo del PDF como contexto** al modelo en cada
pregunta. Aprovechando la amplia ventana de contexto de Gemini, esto resulta más simple,
preciso y económico que un pipeline de embeddings para un documento de este tamaño, y evita
que el modelo invente información fuera del documento.

---

## 🏗️ Arquitectura de la solución

```
                       ┌──────────────────────────────┐
   Usuario  ──HTTP──▶  │        FastAPI (app)         │
 (navegador / API)     │                              │
                       │   GET  /        (chat web)   │
                       │   POST /ask     (pregunta)   │
                       │   GET  /health               │
                       └───────────────┬──────────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼ (al iniciar, 1 vez)    ▼ (por cada pregunta)      │
      ┌───────────────┐        ┌────────────────────┐            │
      │  PDF Loader   │───────▶│  Documento en      │            │
      │  (pypdf)      │ texto  │  memoria (contexto)│◀───────────┘
      └───────────────┘        └─────────┬──────────┘
                                         │ contexto + pregunta
                                         ▼
                               ┌────────────────────┐
                               │   Google Gemini    │
                               │   (gemini-flash)   │
                               └─────────┬──────────┘
                                         ▼
                                Respuesta en lenguaje natural
```

**Flujo:**
1. **Al iniciar:** se lee el PDF y su texto queda cargado en memoria como contexto.
2. **Por cada pregunta:** se envía a Gemini el documento + la pregunta, con instrucciones de
   responder únicamente con base en el documento.
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

### Estructura del repositorio

```
alura-latam/
├── app/
│   ├── main.py          # API FastAPI (endpoints + interfaz)
│   ├── agent.py         # Lógica del agente (carga del PDF + generación con Gemini)
│   ├── pdf_loader.py    # Lectura y limpieza del texto del PDF
│   ├── memory.py        # Memoria de conversaciones en CSV (pandas + filtros)
│   ├── analytics.py     # Análisis de conversaciones (pandas + LLM + JSON)
│   ├── llm.py           # Capa multi-proveedor de LLM (Gemini o Groq)
│   ├── llm_client.py    # Cliente de Google Gemini
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
| IA / LLM         | **Google Gemini** o **Groq** (Llama/Gemma, open source) — conmutable |
| Lectura de PDF   | pypdf                                         |
| Memoria / datos  | pandas (CSV de sesiones, filtros)             |
| Nube / Deploy    | Oracle Cloud Infrastructure (OCI Compute)     |
| Frontend         | HTML + CSS + JavaScript (vanilla)             |
| Testing          | pytest                                        |

---

## 🚀 Instrucciones para ejecutar el proyecto

### 1. Requisitos previos
- Python 3.11 o superior.
- Una **API key de Google Gemini** (gratuita, nivel *free tier*), que se obtiene en
  [Google AI Studio](https://aistudio.google.com/app/apikey).

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

El agente se despliega en una **instancia Compute** de OCI (una VM). El modelo de lenguaje
(Google Gemini) se consume por API, por lo que basta con definir la `GEMINI_API_KEY` en el
servidor. Pasos resumidos:

1. Crear la VM (Ubuntu 22.04) y abrir el puerto `8000` en la *Security List* / *NSG*.
2. Conectarse por SSH y preparar el entorno:
   ```bash
   git clone https://github.com/AndrewCarvajal97/agente-turistico-santander.git
   cd agente-turistico-santander
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env          # edita .env y coloca tu GEMINI_API_KEY
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

_(Las respuestas se generan dinámicamente con Google Gemini a partir del PDF fuente.)_

---

## 📄 Documento fuente

El agente responde a partir de [`data/guia_turistica_santander.pdf`](data/guia_turistica_santander.pdf),
una guía turística de 5 páginas que cubre destinos, deportes de aventura, gastronomía,
transporte, mejor época para visitar y preguntas frecuentes de Santander, Colombia.

---

## 🗺️ Roadmap / próximos pasos

- Migrar a un pipeline **RAG con LangChain** (embeddings + base vectorial) para escalar a
  documentos más grandes o a múltiples fuentes.

---

## 📝 Licencia

Proyecto educativo desarrollado para el Challenge Alura Agente.
