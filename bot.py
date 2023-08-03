import os
import logging
from dotenv import load_dotenv
import aiogram
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, Document
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

if not bot_token:
    raise ValueError("Telegram bot token not found. Please set the TELEGRAM_BOT_TOKEN environment variable.")

bot = Bot(token=bot_token)
dp = Dispatcher(bot)

# Load configuration from config.ini
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Get the values from the configuration
chunk_size = int(config.get("PDF", "chunk_size", fallback="1000"))
chunk_overlap = int(config.get("PDF", "chunk_overlap", fallback="200"))
pdf_size_limit = int(config.get("PDF", "pdf_size_limit", fallback="20"))


user_data = {}


@dp.message_handler(commands=['start'])
async def on_start(message: Message):
    await message.answer("Welcome to the PDF Question Answering Bot! Upload your PDF and ask a question.")


@dp.message_handler(content_types=[types.ContentType.DOCUMENT])
async def on_document_received(message: Message):
    document: Document = message.document

    if document.file_size > pdf_size_limit * 1024 * 1024:
        await message.answer("The uploaded file is too large. Please upload a smaller PDF.")
        return

    # Send a message to indicate that the bot is processing the PDF
    await message.answer("Processing the PDF. This may take a few seconds...")

    try:
        # Get file from Telegram
        file = await bot.get_file(document.file_id)
        await file.download("uploaded_file.pdf")

        # Read the PDF and extract text using pdfplumber
        with pdfplumber.open("uploaded_file.pdf") as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_id = message.from_user.id
        user_data[user_id] = {"knowledge_base": knowledge_base}

        await message.answer("PDF uploaded successfully. You can now ask a question!")
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        await message.answer("An error occurred while processing the PDF. Please try again later.")
    finally:
        # Clean up the downloaded PDF file
        os.remove("uploaded_file.pdf")


@dp.message_handler(content_types=[types.ContentType.TEXT])
async def on_text_message(message: Message):
    user_question = message.text

    user_id = message.from_user.id
    user_info = user_data.get(user_id)
    if user_info is not None:
        knowledge_base = user_info.get("knowledge_base")

        try:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            await message.answer(response)
        except Exception as e:
            logger.error(f"Error while processing the question: {str(e)}")
            await message.answer("An error occurred while processing your question. Please try again later.")
    else:
        await message.answer("Please upload a PDF first.")


def main():
    load_dotenv()
    aiogram.executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    main()
