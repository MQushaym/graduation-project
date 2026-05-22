# PharmaGuide

**PharmaGuide** is an AI-based pharmaceutical guidance system designed to make official drug leaflet information easier to access, search, and understand. It works as a "digital pharmacist" that answers medication-related questions using a Retrieval-Augmented Generation (RAG) pipeline grounded in official Saudi Food and Drug Authority (SFDA) leaflet data.

The system supports multimodal interaction, including text questions, voice input, and image/QR-assisted drug lookup. Its goal is to provide clear, source-grounded pharmaceutical guidance in Arabic and English while reducing the risk of unsupported AI-generated answers.

> **Medical disclaimer:** PharmaGuide is an educational and decision-support tool. It does not replace professional medical advice, diagnosis, treatment, or consultation with a licensed pharmacist or physician.

---

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [How It Works](#how-it-works)
- [Technical Stack](#technical-stack)
- [Dataset and Knowledge Base](#dataset-and-knowledge-base)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Execution](#execution)
- [Operational Notes](#operational-notes)

---

## Features

- **RAG-based medical answering:** Retrieves relevant official drug leaflet content before generating an answer.
- **Text-based questions:** Users can type medication-related questions directly into the interface.
- **Voice input:** Speech is converted to text using Azure Cognitive Services, enabling hands-free interaction.
- **Image/QR-assisted search:** Users can upload or scan supported drug images/QR codes to establish medication context.
- **Arabic and English support:** The knowledge base includes Arabic and English leaflet content, enabling bilingual guidance.
- **Source-grounded responses:** The generation step is constrained by retrieved leaflet context to reduce hallucinations.
- **Persistent vector search:** ChromaDB stores embedded leaflet chunks for fast semantic retrieval.
- **Session isolation:** User state is managed per session to keep concurrent interactions separated.
- **GitHub-ready web interface:** Built with Chainlit and custom public assets for a simple browser-based experience.

---

## Screenshots

### Text Question

<img src="public/assets/%D9%84%D9%82%D8%B7%D8%A9%20%D8%B4%D8%A7%D8%B4%D8%A9%202026-05-17%20222322.png" alt="PharmaGuide text question interface" width="900">

### Image-Based Search

<img src="public/assets/%D9%84%D9%82%D8%B7%D8%A9%20%D8%B4%D8%A7%D8%B4%D8%A9%202026-05-17%20223001.png" alt="PharmaGuide image-based search interface" width="900">

---

## How It Works

PharmaGuide follows a three-stage RAG workflow:

1. **Retrieval:** The user query is embedded and compared against the indexed drug leaflet chunks stored in ChromaDB.
2. **Augmentation:** The most relevant leaflet sections are inserted into the prompt as trusted context.
3. **Generation:** Google Gemini generates an answer using only the retrieved context, with a low-temperature configuration for more stable medical responses.

The core workflow is:

```text
User Input -> Embedding -> Semantic Retrieval -> Context Augmentation -> LLM Response
```

Input-specific processing happens before the shared RAG pipeline:

- **Text input:** The typed question is sent directly to the RAG pipeline.
- **Voice input:** Audio is transcribed first, then processed as a text query.
- **Image/QR input:** The uploaded image is processed to extract drug context before retrieval.

---

## Technical Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| UI | Chainlit + React integration | Browser-based conversational interface |
| Backend | Python | Core application logic and orchestration |
| RAG Orchestration | LangChain | Connects retrieval, prompt construction, and generation |
| Vector Database | ChromaDB | Persistent semantic search over leaflet chunks |
| Embeddings | Voyage AI | Bilingual embedding generation for Arabic and English leaflet content |
| LLM | Google Gemini | Source-grounded answer generation |
| Speech-to-Text | Azure Cognitive Services | Converts voice input into text |
| Computer Vision | OpenCV | QR/image processing and metadata extraction |

---

## Dataset and Knowledge Base

The knowledge base is built from structured pharmaceutical leaflet data sourced from the Saudi Drug Information System (SDIS), administered by the Saudi Food and Drug Authority (SFDA).

Key dataset characteristics:

- **Source:** Saudi Drug Information System (SDIS)
- **Raw entries:** 8,500+ drug records
- **Filtered entries:** Approximately 7,300 records after removing entries without official Patient Information Leaflets
- **Format:** Structured JSON files
- **Content:** Drug metadata, English PIL text, Arabic PIL text, and Summary of Product Characteristics
- **Chunking strategy:** Long leaflet text is split into large overlapping chunks to preserve medical context
- **Vectorization:** Chunks are embedded and stored in ChromaDB for semantic retrieval

This design allows the system to update the knowledge base without retraining the language model.

---

## Evaluation

PharmaGuide was evaluated using a specialized validation dataset of **20 high-fidelity records** representing realistic pharmaceutical questions in the Saudi market. Each record maps a user question to a human-verified ground-truth answer sourced from official SFDA documents.

The evaluation used the **RAGAS** framework to measure both retrieval quality and answer generation reliability.

| Metric | Score | Interpretation |
| --- | ---: | --- |
| Faithfulness | 0.8750 | High factual consistency with minimal hallucination |
| Context Recall | 0.9814 | Strong ability to retrieve the correct leaflet context |
| Answer Relevancy | 0.9464 | Answers directly address user questions |
| Answer Correctness | 0.9862 | High agreement with verified ground-truth answers |
| Answer Similarity | 0.8491 | Strong semantic alignment with reference answers |

These results indicate that PharmaGuide provides accurate, source-grounded guidance and that its retrieval component consistently finds the relevant medical information needed for answer generation.

---

## Project Structure

The project is organized into modular directories to separate configuration, public assets, source code, and persistent data:

```text
.
|-- .chainlit/           # Chainlit configuration and localization files
|-- data/
|   `-- ChromaDB/        # Persisted vector database
|-- public/              # Static assets, CSS, and JavaScript
|   |-- assets/          # Images, logos, and interface screenshots
|   |-- css/             # Custom styling
|   `-- js/              # Client-side scripts
|-- src/                 # Source code
|   |-- logic.py         # Core engine: vision, speech, and RAG processors
|   `-- main.py          # Application entry point and UI orchestration
|-- .env                 # Environment variables
|-- .gitignore           # Version control exclusions
`-- requirements.txt     # Production dependencies
```

---

## Setup and Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Google Gemini Configuration
GOOGLE_API_KEY=your_google_gemini_api_key

# Voyage AI Configuration
VOYAGE_API_KEY=your_voyage_api_key

# Azure Speech Services Configuration
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_REGION=your_azure_service_region
```

---

## Execution

Run the application from the project root:

```bash
python -m chainlit run src/main.py
```

After startup, Chainlit will provide a local browser URL in the terminal.

---

## Operational Notes

- Run commands from the project root so relative paths resolve correctly.
- Keep `.env` out of version control because it contains private API keys.
- Ensure the `data/ChromaDB` directory is available or regenerated before running retrieval.
- For best precision, provide a medication context through text, voice, or supported image/QR input before asking detailed drug-specific questions.
- Generated responses should be reviewed by a healthcare professional before being used for clinical decisions.
