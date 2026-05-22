# Technical Implementation: RAG-Based Medical Assistant

This repository contains an integrated medical information retrieval system. The architecture utilizes **Retrieval-Augmented Generation (RAG)** to provide grounded responses based on official medication leaflets. The system incorporates **Computer Vision** for metadata extraction via QR codes and **Speech-to-Text** processing for multi-modal user interaction.

## Key Technical Stack
* **Orchestration:** LangChain
* **User Interface:** React (integrated via Chainlit)
* **Vector Database:** ChromaDB
* **Embeddings:** VoyageAI
* **Large Language Model:** Google Gemini (Generative AI)
* **Speech Processing:** Azure Cognitive Services
* **Computer Vision:** OpenCV

---

## Project Structure
The project is organized into modular directories to separate logic, public assets, and data persistence:

```
.
├── .chainlit/           # Configuration and localization files
├── data/
│   └── ChromaDB/        # Persisted Vector Database
├── public/              # Static assets, CSS, and JavaScript
│   ├── assets/          # Images and logos
│   ├── css/             # Custom styling
│   └── js/              # Client-side scripts
├── src/                 # Source code
│   ├── logic.py         # Core Engine (Vision, Speech, and RAG Processors)
│   └── main.py          # Application entry point and UI orchestration
├── .env                 # Environment variables (Required)
├── .gitignore           # Version control exclusions
└── requirements.txt     # Production dependencies

```


## Setup and Installation

### 1. Environment Configuration

Create a `.env` file in the root directory. This file is mandatory for authenticating with the integrated cloud services. Populate it with the following keys:

```
# Google Gemini Configuration
GOOGLE_API_KEY=your_google_gemini_api_key

# VoyageAI Configuration
VOYAGE_API_KEY=your_voyage_api_key

# Azure Speech Services Configuration
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_REGION=your_azure_service_region

```

### 2. Dependency Installation

Ensure you have Python installed. It is recommended to use a virtual environment. Install the required libraries using the following command:

```
pip install -r requirements.txt

```


## Execution

To launch the application, navigate to the project root directory in your terminal and execute the following command:

```
python -m chainlit run src/main.py

```

The system will initialize the vector database and the cloud service connectors. Once the local server is running, the interface will be accessible via the default local URL provided in the terminal output.

## Operational Notes

* **Execution Path:** All execution commands must be run from the root directory to ensure correct relative path resolution for the `src` and `public` folders.
* **State Management:** The system utilizes `cl.user_session` to ensure state isolation between concurrent users.
* **Context Grounding:** For optimal precision, it is recommended to provide a QR code to establish the medication context before initiating queries.

