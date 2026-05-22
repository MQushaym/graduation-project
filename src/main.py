import os
import asyncio
import chainlit as cl
from logic import PharmaEngine

# Initialize the global engine (Stateless processors are safe globally)
engine = PharmaEngine()

@cl.on_chat_start
async def start():
    """تهيئة الجلسة وبث رسالة الترحيب الرسمية"""
    # Initialize secure user session states
    cl.user_session.set("current_docs", None)
    cl.user_session.set("audio_buffer", bytearray())
    
    welcome_content = """### PharmaGuide Intelligent System
**المساعد التقني الموحد للإرشاد الدوائي**

* يرجى إدراج صورة رمز الاستجابة السريع (QR Code) الخاص بالدواء لتثبيت سياق البيانات.
* يمكنكم طرح الاستفسارات نصياً أو عبر التسجيل الصوتي لمباشرة تحليل النشرات الطبية."""
    
    await cl.Message(content=welcome_content).send()

async def process_text_query(query_text: str):
    """Orchestrates the RAG processing block with Real-time Streaming"""
    current_docs = cl.user_session.get("current_docs")
    
    # Initialize an empty message for streaming
    msg = cl.Message(content="")
    await msg.send()

    try:
        if current_docs:
            stream = engine.rag.get_answer_from_context(current_docs, query_text)
        else:
            stream = engine.rag.get_answer(query_text)
        
        # Stream the response word-by-word
        async for chunk in stream:
            await msg.stream_token(chunk)
            
        # Finalize the message update once streaming is complete
        await msg.update()
        
    except asyncio.CancelledError:
        # Standard behavior if user disconnects mid-generation
        raise 
    except Exception as e:
        msg.content += f"\n\n**خطأ تقني في المعالجة:** لا يمكن إتمام الطلب حالياً. التفاصيل: {str(e)}"
        await msg.update()

@cl.on_message
async def main(message: cl.Message):
    # 1. Visual Data Processing (QR Codes)
    if message.elements:
        for element in message.elements:
            if "image" in element.mime:
                status_msg = cl.Message(content="جاري استخراج بيانات الترميز...")
                await status_msg.send()
                
                try:
                    reg_num = engine.vision.extract_qr_data(element.path)
                    if reg_num:
                        docs = engine.rag.get_docs_by_registration(reg_num)
                        if docs:
                            cl.user_session.set("current_docs", docs)
                            info = engine.rag.format_docs([docs[0]])
                            await cl.Message(content=f"تم إثبات سياق الدواء بنجاح:\n{info}").send()
                        else:
                            await cl.Message(content=f"رقم التسجيل ({reg_num}) غير مدرج في قاعدة البيانات.").send()
                    else:
                        await cl.Message(content="لم يتم العثور على رمز استجابة سريع (QR Code) صالح في الصورة.").send()
                except ValueError as ve:
                    await cl.Message(content=f"تعذر معالجة الصورة: {str(ve)}").send()
                except Exception as e:
                    await cl.Message(content="حدث خطأ غير متوقع أثناء معالجة الصورة.").send()
                finally:
                    await status_msg.remove()
                    
                if not message.content: return

    # 2. Text Query Processing
    if not message.content or message.content.strip() == "":
        return
        
    await process_text_query(message.content)

# --- Isolated Audio Streaming Protocol ---

@cl.on_audio_start
async def on_audio_start():
    # Reset specific user's buffer
    cl.user_session.set("audio_buffer", bytearray())
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    # Append to isolated session buffer
    buffer = cl.user_session.get("audio_buffer")
    if buffer is not None:
        buffer.extend(chunk.data)

@cl.on_audio_end
async def on_audio_end():
    buffer = cl.user_session.get("audio_buffer")
    
    # Validation: Prevent malformed headers from 0-byte or micro-second audio
    if not buffer or len(buffer) < 4000:
        await cl.Message(content="المقطع الصوتي قصير جداً لتكوين إشارة واضحة. يرجى إعادة المحاولة.").send()
        return

    temp_wav = None
    transcription_status = cl.Message(content="جاري تحليل الإشارة الصوتية...")
    await transcription_status.send()

    try:
        # Save securely with UUID
        temp_wav = engine.speech.save_audio_file(buffer)
        recognized_text = await engine.speech.transcribe_audio(temp_wav)
        
        if recognized_text:
            await transcription_status.remove()
            # Push cleanly directly to the text processing pipeline
            await process_text_query(recognized_text)
        else:
            transcription_status.content = "تعذر تحليل المحتوى الصوتي. الإشارة غير واضحة."
            await transcription_status.update()
            
    except Exception as e:
        transcription_status.content = "حدث انقطاع في الخدمة السحابية للصوتيات."
        await transcription_status.update()
        
    finally:
        # Atomic Cleanup: Guarantees file is purged even if the cloud API times out
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass # Already deleted or locked by OS (graceful fail)