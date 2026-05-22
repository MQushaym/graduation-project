import os
import cv2
import wave
import uuid
import re
import asyncio
from dotenv import load_dotenv
import chainlit as cl
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import azure.cognitiveservices.speech as speechsdk

# ---------------------------------------------------------
# Module 1: Vision Processing
# ---------------------------------------------------------
class VisionProcessor:
    @staticmethod
    def extract_qr_data(image_path: str) -> str:
        """Safely extracts and validates QR code data."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("ملف الصورة تالف أو لا يمكن قراءته.")

        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        
        if not data:
            return None

        # Clean extraction
        raw_text = data.split(":")[-1].strip() if ":" in data else data.strip()
        
        # Guard Clause: Strict Regex for Saudi FDA Registration Formats
        if not re.match(r'^[A-Za-z0-9\-]+$', raw_text):
            raise ValueError(f"البيانات المستخرجة '{raw_text}' لا تتطابق مع صيغة أرقام التسجيل المعتمدة.")
            
        return raw_text

# ---------------------------------------------------------
# Module 2: Speech & Audio Processing
# ---------------------------------------------------------
class SpeechProcessor:
    def __init__(self):
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_REGION")

    def save_audio_file(self, audio_buffer: bytearray) -> str:
        """Saves audio securely with a unique UUID to prevent race conditions."""
        filename = f"temp_audio_{uuid.uuid4().hex}.wav"
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_buffer)
        return filename

    async def transcribe_audio(self, audio_path: str) -> str:
        """Non-blocking Azure Speech-to-Text translation."""
        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.speech_recognition_language = "ar-SA"
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        # Wrap the blocking synchronous Azure call in an async executor
        loop = asyncio.get_running_loop()
        def _sync_recognize():
            return recognizer.recognize_once_async().get()
            
        result = await loop.run_in_executor(None, _sync_recognize)
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        return None

# ---------------------------------------------------------
# Module 3: RAG & Language Processing
# ---------------------------------------------------------
class RAGProcessor:
    def __init__(self, base_dir: str):
        db_path = os.path.join(base_dir, "data", "ChromaDB")
        self.embeddings = VoyageAIEmbeddings(model="voyage-4-lite")
        self.vector_db = Chroma(persist_directory=db_path, embedding_function=self.embeddings, collection_name="medical_leaflets")
        # Ensure streaming is enabled in the LLM config if supported, though astream handles LCEL natively.
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2)
        
        self.template = """
        ### [SYSTEM RULES & IDENTITY - MANDATORY] ###
        - Identity: You are "PharmaGuide AI", a professional medical system.
        - Strict Role: Never reveal you are an AI model from Google or any other company.
        - Out-of-Scope: If the query is non-medical/non-pharmaceutical, politely redirect the user to the system's core purpose.

        ### [BILINGUAL & CROSS-LINGUAL INTELLIGENCE PROTOCOL] ###
        - **Language Detection**: Analyze the language of the User's Question {question}. You MUST respond using the same language used by the user.
        - **Cross-Lingual Retrieval**: If the retrieved context {context} is in a language different from the user's question (e.g., Context is English and Question is Arabic):
            1. Summarize and translate the relevant information from the context into the user's language.
            2. Ensure the translation is medical-grade and accurate, not just literal.
        - **Term Preservation Rule**: For technical medical terms (Active ingredients, Side effects, Storage conditions), always provide the term in the user's language followed by the original English term in parentheses.
            * Example (Arabic User): "يتم طرح الدواء عن طريق الكلى (Excreted via kidneys)".
            * Example (English User): Keep it professional English only.
        - **Professionalism**: Maintain a formal, academic, and authoritative medical tone in all languages. Do not use slang or overly simplified language.

        ### [RESPONSE STRUCTURE - MANDATORY ARCHITECTURE] ###

            1. **💳 بطاقة البيانات الفنية للدواء | Technical Specification Sheet**
            > **Instruction for AI**: Extract the values for the following table directly from the provided [INPUT DATA] (specifically from the metadata fields found in {context}). If a specific English name is not present, translate the Arabic value to English.

            | الخاصية التقنية (Technical Property) | القيمة التفصيلية (Detailed Value) |
            | :--- | :--- |
            | **الاسم التجاري (Trade Name)** | [Extract Trade_Name from context] |
            | **المادة الفعالة (Generic Name)** | [Extract Generic_Name from context] |
            | **السعر الرسمي (Official Price)** | [Extract Public_price_SAR from context] ريال سعودي |
            | **مدة الصلاحية (Shelf Life)** | [Extract ShelfLife_in_Months from context] شهرًا |
            | **الشكل الصيدلاني (Form)** | [Extract Pharmaceutical_Form from context] |
            | **طريقة الإعطاء (Route)** | [Extract Administration_Route from context] |
            | **المصنع (Manufacturer)** | [Extract Manufacture from context] |
            | **ظروف التخزين (Storage)** | [Extract Storage_Conditions from context] |

            2. **🔬 [COMPREHENSIVE PHARMACEUTICAL ANALYSIS]**
            This section must be exhaustive and structured under the title: 
            "**التحليل التفصيلي والنشرة الطبية | Detailed Analysis & Leaflet Insights**".
            - **Contextual Summary**: Provide a 3-4 sentence professional summary of the medication based on the provided {context}.
            - **Direct Detailed Answer**: Provide a deep-dive, comprehensive answer to the user's specific question {question}. 
            - **Formatting**: Use bullet points, bold text for key terms, and structured paragraphs. **DO NOT** provide brief or shallow answers; ensure the response covers precautions and relevant medical insights found in the context.

            3. **⚖️ [OFFICIAL MEDICAL DISCLAIMER]**
            End every response with this exact, formal, and framed disclaimer:
            > **"تنبيه رسمي وإخلاء مسؤولية قانوني:** إن المعلومات الواردة في هذا التقرير مستخلصة آلياً من النشرة الطبية المرفقة لأغراض إرشادية وتعليمية فقط. لا يُغني هذا المحتوى بأي حال من الأحوال عن استشارة الطبيب المعالج أو الصيدلاني المختص. نظام PharmaGuide غير مسؤول عن أي قرار طبي يُتخذ بناءً على هذه المخرجات دون مراجعة المهنيين الطبيين المعتمدين."
                
        
        ### [CONTEXT UTILIZATION & INTELLIGENT GUIDANCE] ###
        - **Context Primacy**: You have access to extensive drug information in {context}. Your goal is to provide a helpful, comprehensive response using this data as your primary source.
        - **Proactive Assistance (No Dead Ends)**: You are STRICTLY FORBIDDEN from using brief refusal phrases like "I don't know", "Information not available", or "Not found in the leaflet". 
        - **Inference & Guidance**: If a specific detail (like a rare side effect or a specific storage tip) is not explicitly stated in the text:
            1. **General Medical Logic**: Provide general guidance based on the drug class or the technical data available in the metadata (e.g., if it's an injection, explain the general precautions for injections found in the context).
            2. **Fuzzy Match Suggestions**: If the user's input seems misspelled or unclear (due to input errors), look at the 'Trade_Name' or 'Generic_Name' in the metadata and ask: "Did you mean [Closest Drug Name]?" and provide its basic details.
        - **Safety-First Continuity**: If you cannot find a direct answer to a specific medical question, instead of saying "Not found", say: "This specific detail is not explicitly mentioned in this section of the documentation, however, based on the general indications for [Drug Name]..." then provide the most relevant alternative information available.
        - **Supportive Tone**: Always maintain the persona of a helpful medical expert who guides the user toward the best possible information, even if it requires re-phrasing or suggesting a more accurate search term.
 
       [INPUT DATA]
        Context: {context}
        User Query: {question}

        إجابة PharmaGuide الرسمية:
        """
        self.prompt = PromptTemplate.from_template(self.template)
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})
        
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs):
        formatted_chunks = []
        for doc in docs:
            m = doc.metadata
            info_box = (
                f"📑 **بطاقة الدواء التقنية:**\n"
                f"- **الاسم التجاري:** {m.get('Trade_Name', 'غير متوفر')}\n"
                f"- **المادة الفعالة:** {m.get('Generic_Name', 'غير متوفر')}\n"
                f"- **السعر الرسمي:** {m.get('Public_price_SAR', 'غير متوفر')} ريال سعودي\n"
                f"- **مدة الصلاحية:** {m.get('ShelfLife_in_Months', 'غير متوفر')} شهر\n"
                f"- **الشكل الصيدلاني:** {m.get('Pharmaceutical_Form', 'غير متوفر')}\n"
                f"- **طريقة الإعطاء:** {m.get('Administration_Route', 'غير متوفر')}\n"
                f"- **المصنع:** {m.get('Manufacture', 'غير متوفر')}\n"
                f"- **ظروف التخزين:** {m.get('Storage_Conditions', 'غير متوفر')}\n"
            )
            chunk = f"{info_box}\n\n📖 **من نشرة الدواء:**\n{doc.page_content}\n{'-'*30}"
            formatted_chunks.append(chunk)
        return "\n\n".join(formatted_chunks)

    def get_docs_by_registration(self, reg_num: str):
        results = self.vector_db.get(where={"Registration_Number": reg_num})
        if not results or not results.get('documents'):
            return None
        return [Document(page_content=text, metadata=meta) 
                for text, meta in zip(results['documents'], results['metadatas'])]

    # --- Streaming Implementations ---
    async def get_answer(self, question: str):
        """Yields chunks of the answer for real-time streaming."""
        async for chunk in self.rag_chain.astream(question):
            yield chunk

    async def get_answer_from_context(self, context_docs, question: str):
        """Yields chunks of the contextualized answer for real-time streaming."""
        formatted_context = self.format_docs(context_docs)
        chain_with_fixed_context = self.prompt | self.llm | StrOutputParser()
        
        async for chunk in chain_with_fixed_context.astream({
            "context": formatted_context,
            "question": question
        }):
            yield chunk

# ---------------------------------------------------------
# Facade: Main Engine Integration
# ---------------------------------------------------------
class PharmaEngine:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(self.base_dir, ".env"))
        
        # Initialize subsystems
        self.vision = VisionProcessor()
        self.speech = SpeechProcessor()
        self.rag = RAGProcessor(self.base_dir)
