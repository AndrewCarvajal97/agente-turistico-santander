# 🏔️ Agente Turístico de Santander — Challenge Alura Agente

Agente de inteligencia artificial que responde preguntas sobre los **sitios turísticos
del departamento de Santander, Colombia**, a partir del contenido de un documento PDF.
Utiliza la técnica **RAG (Retrieval-Augmented Generation)** con **Google Gemini**
(chat + embeddings), está expuesto mediante una **API con FastAPI** con interfaz web de
chat, y se despliega en **Oracle Cloud Infrastructure (OCI Compute)**.

> Proyecto desarrollado para el **Challenge Alura Agente**.

---

## 📖 Descripción general

El agente permite a un usuario hacer preguntas en lenguaje natural (por ejemplo,
*"¿dónde puedo practicar rafting?"*) y obtener respuestas fundamentadas exclusivamente en
una guía turística en formato PDF. En lugar de enviar el documento completo al modelo en
cada pregunta, el sistema:

1. **Indexa** el PDF una sola vez (lo divide en fragmentos y genera *embeddings*).
2. En cada consulta, **recupera** los fragmentos más relevantes por similitud semántica.
3. **Genera** la respuesta con un modelo de lenguaje (LLM), usando solo ese contexto.

Esto hace las respuestas más precisas, económicas y libres de alucinaciones.

---

## 🏗️ Arquitectura de la solución

```
                         ┌──────────────────────────────┐
   Usuario  ──HTTP──▶    │        FastAPI (app)         │
 (navegador / API)       │                              │
                         │   GET  /        (chat web)   │
                         │   POST /ask     (pregunta)   │
                         │   POST /reindex              │
                         │   GET  /health               │
                         └───────────────┬──────────────┘
                                         │
        ┌────────────────────────────────┼───────────────────────────────┐
        │                                 │                               │
        ▼ (indexación, 1 vez)             ▼ (por pregunta)                │
 ┌─────────────┐   ┌──────────┐    ┌────────────────┐   ┌──────────────┐   │
 │  PDF Loader │──▶│ Chunker  │──▶ │   Embeddings   │──▶│ Vector Store │◀──┘
 │  (pypdf)    │   │ (overlap)│    │ (Gemini embed) │   │ (NumPy/coseno)│
 └─────────────┘   └──────────┘    └────────────────┘   └──────┬───────┘
                                                              │ top-k
                                                              ▼
                                                    ┌──────────────────┐
                                                    │  Google Gemini   │
                                                    │    LLM (chat)    │
                                                    └────────┬─────────┘
                                                             ▼
                                                     Respuesta + fuentes
```

**Flujo RAG:**
- **Indexación:** `pdf_loader` → `chunker` → `embeddings` → `vector_store` (se guarda en `data/index.npz`).
- **Consulta:** pregunta → *embedding* → búsqueda por coseno → contexto → LLM → respuesta.

### Estructura del repositorio

```
alura-latam/
├── app/
│   ├── main.py          # API FastAPI (endpoints + interfaz)
│   ├── agent.py         # Orquestación del RAG (indexar / preguntar)
│   ├── pdf_loader.py    # Lectura y limpieza del PDF
│   ├── chunker.py       # Fragmentación con solapamiento
│   ├── embeddings.py    # Embeddings con OCI Generative AI
│   ├── vector_store.py  # Índice vectorial NumPy (similitud del coseno)
│   ├── llm_client.py    # Cliente de Google Gemini
│   └── config.py        # Configuración desde variables de entorno
├── static/index.html    # Interfaz web de chat
├── scripts/build_index.py  # Construye el índice por CLI
├── tests/test_agent.py  # Tests unitarios (sin OCI)
├── data/
│   ├── guia_turistica_santander.pdf   # Documento fuente
│   └── guia_turistica_santander.md    # Versión editable
├── docs/                # Diagramas y capturas del deploy
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## 🛠️ Tecnologías y herramientas

| Categoría          | Herramienta                                   |
|--------------------|-----------------------------------------------|
| Lenguaje           | Python 3.11+                                  |
| API web            | FastAPI + Uvicorn                             |
| IA / LLM           | Google Gemini — `gemini-2.0-flash`            |
| Embeddings         | Google Gemini — `text-embedding-004`          |
| Búsqueda vectorial | NumPy (similitud del coseno)                  |
| Lectura de PDF     | pypdf                                         |
| Nube / Deploy      | Oracle Cloud Infrastructure (OCI Compute)     |
| Frontend           | HTML + CSS + JavaScript (vanilla)             |
| Testing            | pytest                                        |

---

## 🚀 Instrucciones para ejecutar el proyecto

### 1. Requisitos previos
- Python 3.11 o superior.
- Una **API key de Google Gemini** (gratuita), que se obtiene en
  [Google AI Studio](https://aistudio.google.com/app/apikey).
- (Para el deploy) una cuenta de **Oracle Cloud (OCI)** con acceso a **Compute**.

### 2. Clonar e instalar dependencias
```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
```bash
cp .env.example .env
```
Edita `.env` y completa:
- `GEMINI_API_KEY` — tu clave de [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Construir el índice del documento
```bash
python scripts/build_index.py
```

### 5. Levantar la aplicación
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Abre tu navegador en **http://localhost:8000** para usar el chat, o consulta la documentación
interactiva de la API en **http://localhost:8000/docs**.

### 6. (Opcional) Ejecutar los tests
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
   git clone https://github.com/<tu-usuario>/<tu-repo>.git
   cd <tu-repo>
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env          # edita .env y coloca tu GEMINI_API_KEY
   python scripts/build_index.py
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. (Opcional) Contenerizar con el `Dockerfile` incluido y correr con Docker.

> 🔗 **Aplicación desplegada:** _(añade aquí la URL pública, ej. `http://<IP-pública>:8000`)_
> 🖼️ **Captura del deploy:** ver `docs/captura-deploy.png`.

---

## 💬 Ejemplos de preguntas que el agente puede responder

- ¿Cuál es la capital de Santander?
- ¿Cuál es el principal atractivo natural del departamento?
- ¿Dónde puedo practicar rafting y en qué ríos?
- ¿Por qué Barichara es tan famosa? ¿Qué es el Camino Real?
- ¿Qué comida típica debo probar en Santander?
- ¿Cuánto se tarda de Bucaramanga a San Gil?
- ¿Cuál es la mejor época para visitar?
- ¿Qué deportes extremos puedo practicar?

## 🗨️ Ejemplos de respuestas generadas

> **Pregunta:** ¿Dónde puedo practicar rafting?
>
> **Respuesta:** Puedes practicar rafting en **San Gil**, considerada la capital del turismo
> de aventura de Colombia. El **río Fonce** es ideal para principiantes (grado II-III),
> mientras que el **río Suárez** ofrece un nivel avanzado (grado IV-V) para personas con
> experiencia.

> **Pregunta:** ¿Por qué Barichara es tan famosa?
>
> **Respuesta:** Barichara es considerada *"el pueblo más lindo de Colombia"*. Es un
> **Monumento Nacional** reconocido por su arquitectura colonial: calles empedradas, casas
> blancas de tapia pisada y techos de teja de barro. Desde allí parte el **Camino Real**, un
> sendero histórico de unos 5,3 km que conecta con el pueblo de Guane.

_(Las respuestas se generan dinámicamente con OCI Generative AI a partir del PDF fuente.)_

---

## 📄 Documento fuente

El agente responde a partir de [`data/guia_turistica_santander.pdf`](data/guia_turistica_santander.pdf),
una guía turística de 5 páginas que cubre destinos, deportes de aventura, gastronomía,
transporte, mejor época para visitar y preguntas frecuentes de Santander, Colombia.

---

## 📝 Licencia

Proyecto educativo desarrollado para el Challenge Alura Agente.
