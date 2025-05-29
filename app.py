import gradio as gr
import os
from load_and_embed import load_vectorstore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 初期化（最初の1回のみ）
vectorstore = load_vectorstore()

def answer_question(query):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not not openai_api_key:
        return "❌ OpenAI APIキーが設定されていません。"
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    result = qa_chain.run(query)
    return result

demo = gr.Interface(fn=answer_question, 
                    inputs=gr.Textbox(lines=2, placeholder="質問を入力してください"), 
                    outputs="text",
                    title="📚 あなたの書籍チャットボット",
                    description="あなたの本の内容に基づいてAIが答えます。")

demo.launch()