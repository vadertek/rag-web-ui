import json
import base64
import re
from typing import List, AsyncGenerator, Any
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings
from app.models.chat import Message
from app.models.knowledge import KnowledgeBase, Document
from langchain.globals import set_verbose, set_debug
from app.services.vector_store import VectorStoreFactory
from app.services.embedding.embedding_factory import EmbeddingsFactory
from app.services.llm.llm_factory import LLMFactory

set_verbose(True)
set_debug(True)

async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    db: Session
) -> AsyncGenerator[str, None]:
    try:
        # Create user message
        user_message = Message(
            content=query,
            role="user",
            chat_id=chat_id
        )
        db.add(user_message)
        db.commit()
        
        # Create bot message placeholder
        bot_message = Message(
            content="",
            role="assistant",
            chat_id=chat_id
        )
        db.add(bot_message)
        db.commit()
        
        # Get knowledge bases and their documents
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        
        # Initialize embeddings
        embeddings = EmbeddingsFactory.create()
        
        # Create a vector store for each knowledge base
        vector_stores = []
        for kb in knowledge_bases:
            documents = db.query(Document).filter(Document.knowledge_base_id == kb.id).all()
            if documents:
                # Use the factory to create the appropriate vector store
                vector_store = VectorStoreFactory.create(
                    store_type=settings.VECTOR_STORE_TYPE,  # 'chroma' or other supported types
                    collection_name=f"kb_{kb.id}",
                    embedding_function=embeddings,
                )
                print(f"Collection {f'kb_{kb.id}'} count:", vector_store._store._collection.count())
                vector_stores.append(vector_store)
        
        if not vector_stores:
            error_msg = "I don't have any knowledge base to help answer your question."
            yield f'0:"{error_msg}"\n'
            yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'
            bot_message.content = error_msg
            db.commit()
            return
        
        # Use first vector store for now
        retriever = vector_stores[0].as_retriever()
        
        # Initialize the language model
        llm = LLMFactory.create()
        
        # Create contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Create history aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, 
            retriever,
            contextualize_q_prompt
        )

        print("contextualize_q_prompt:", contextualize_q_prompt)

        # Create QA prompt
        qa_system_prompt = (
            "You are given a user question and a set of related contexts. Your task is to write a clean, concise, and accurate answer, "
            "written by an expert in an unbiased and professional tone. The contexts are numbered in order, starting from 1. "
            "Cite sources using the format [citation:x] at the end of each relevant sentence, where x refers to the context number. "
            "If a sentence is based on multiple contexts, cite all relevant ones (e.g., [citation:1][citation:2]).\n\n"
            "Only include information supported by the provided contexts. If information is missing, write 'information is missing on' followed by the topic. "
            "Do not repeat context verbatim. Limit your response to 1024 tokens. Do not include unrelated details.\n\n"
            "Context: {context}\n\n"
            "Important:\n"
            "- Your answer must be written in the same language as the question (Ukrainian, English, etc.), including product names where applicable.\n"
            "- If the question is in Ukrainian, use the Ukrainian translations from the product dictionary below when referring to TropiClean products.\n"
            "- Always include the product name when relevant, explain its usefulness, and cite the source context.\n"
            "PRODUCT TRANSLATIONS DICTIONARY:\n"
            "Oral Care Gels:\n"
            "• TropiClean Fresh Breath Oral Care Brushing Gel for Dogs → Тропіклеан Чудернадський Гель для циркової собаки\n"
            "• TropiClean Fresh Breath Oral Care Gel for Dogs → Тропіклеан Освіжаючий Гель для псів з диханням дракона\n"
            "• TropiClean Fresh Breath Oral Care Gel for Cats → Тропіклеан Гель для котів, які дихають як трактор\n"
            "• TropiClean Fresh Breath Oral Care Gel for Puppies → Тропіклеан Гель для щенят, що гризуть усе на світі\n"
            "• TropiClean Fresh Breath Oral Care Gel for Dogs – Peanut Butter Flavor → Тропіклеан Арахісовий Гель для гурманів на чотирьох лапах\n\n"
            "Shampoos:\n"
            "• TropiClean Aloe & Coconut Deodorizing Shampoo → Тропіклеан Шампунь для псів із запахом пляжного курорту\n"
            "• TropiClean Awapuhi & Coconut Whitening Shampoo → Тропіклеан Відбілюючий Шампунь для тих, хто хоче сяяти як лампочка\n"
            "• TropiClean Berry & Coconut Deep Cleansing Shampoo → Тропіклеан Ягідно-Кокосовий Шампунь для найзаляпаніших хвостатих\n"
            "• TropiClean Gentle Coconut Hypoallergenic Shampoo for Puppies & Kittens → Тропіклеан Ніжний Шампунь для пухнастиків, які нюхають усе підряд\n"
            "• TropiClean Lime & Coconut Shed Control Shampoo → Тропіклеан Шампунь проти линьки для пухнастиків, що засмічують пилосос\n"
            "• TropiClean Neem & Citrus Flea & Tick Relief Shampoo for Dogs → Тропіклеан Цитрусовий Антиблошиний Шампунь для псів-вояків\n"
            "• TropiClean Oatmeal & Tea Tree Medicated Itch Relief Shampoo → Тропіклеан Вівсяний Шампунь для чухачів зі стажем\n\n"
            "Note: Apply translations only if the user question is in Ukrainian."
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # 修改 create_stuff_documents_chain 来自定义 context 格式
        document_prompt = PromptTemplate.from_template("\n\n- {page_content}\n\n")

        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
            document_variable_name="context",
            document_prompt=document_prompt
        )

        # Create retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )

        # Generate response
        chat_history = []
        for message in messages["messages"]:
            if message["role"] == "user":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                # if include __LLM_RESPONSE__, only use the last part
                if "__LLM_RESPONSE__" in message["content"]:
                    message["content"] = message["content"].split("__LLM_RESPONSE__")[-1]
                chat_history.append(AIMessage(content=message["content"]))

        full_response = ""
        async for chunk in rag_chain.astream({
            "input": query,
            "chat_history": chat_history
        }):
            if "context" in chunk:
                serializable_context = []
                for context in chunk["context"]:
                    serializable_doc = {
                        "page_content": context.page_content.replace('"', '\\"'),
                        "metadata": context.metadata,
                    }
                    serializable_context.append(serializable_doc)
                
                # 先替换引号，再序列化
                escaped_context = json.dumps({
                    "context": serializable_context
                })

                # 转成 base64
                base64_context = base64.b64encode(escaped_context.encode()).decode()

                # 连接符号
                separator = "__LLM_RESPONSE__"
                
                yield f'0:"{base64_context}{separator}"\n'
                full_response += base64_context + separator

            if "answer" in chunk:
                answer_chunk = chunk["answer"]
                full_response += answer_chunk
                # Escape quotes and use json.dumps to properly handle special characters
                escaped_chunk = (answer_chunk
                    .replace('"', '\\"')
                    .replace('\n', '\\n'))
                yield f'0:"{escaped_chunk}"\n'
            
        # Update bot message content
        bot_message.content = full_response
        db.commit()
            
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        yield '3:{text}\n'.format(text=error_message)
        
        # Update bot message with error
        if 'bot_message' in locals():
            bot_message.content = error_message
            db.commit()
    finally:
        db.close()

def generate_response_sync(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    db: Session
) -> Any:
    try:
        # Create user message
        user_message = Message(
            content=query,
            role="user",
            chat_id=chat_id
        )
        db.add(user_message)
        db.commit()
        
        # Create bot message placeholder
        bot_message = Message(
            content="",
            role="assistant",
            chat_id=chat_id
        )
        db.add(bot_message)
        db.commit()
        
        # Get knowledge bases and their documents
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        
        # Initialize embeddings
        embeddings = EmbeddingsFactory.create()
        
        # Create a vector store for each knowledge base
        vector_stores = []
        for kb in knowledge_bases:
            documents = db.query(Document).filter(Document.knowledge_base_id == kb.id).all()
            if documents:
                # Use the factory to create the appropriate vector store
                vector_store = VectorStoreFactory.create(
                    store_type=settings.VECTOR_STORE_TYPE,  # 'chroma' or other supported types
                    collection_name=f"kb_{kb.id}",
                    embedding_function=embeddings,
                )
                print(f"Collection {f'kb_{kb.id}'} count:", vector_store._store._collection.count())
                vector_stores.append(vector_store)
        
        if not vector_stores:
            error_msg = "I don't have any knowledge base to help answer your question."
            bot_message.content = error_msg
            db.commit()
            return error_msg
        
        # Use first vector store for now
        retriever = vector_stores[0].as_retriever()
        
        # Initialize the language model
        llm = LLMFactory.create()
        
        # Create contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Create history aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, 
            retriever,
            contextualize_q_prompt
        )

        # Create QA prompt
        qa_system_prompt = (
            "You are given a user question, and please write a clean, concise, and accurate answer to the question. "
            "You will be given a set of related contexts to the question, which are numbered sequentially starting from 1. "
            "Each context has an implicit reference number based on its position in the array (first context is 1, second is 2, etc.). "
            "Please use these contexts and cite them using the format [citation:x] at the end of each sentence where applicable. "
            "Your answer must be correct, accurate, and written by an expert using an unbiased and professional tone. "
            "Please limit your response to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. "
            "Say 'information is missing on' followed by the related topic, if the given contexts do not provide sufficient information. "
            "If a sentence draws from multiple contexts, list all applicable citations, like [citation:1][citation:2]. "
            "Other than code, specific names, and citations, your answer must be written in the same language as the question. "
            "Be concise. \n\nContext: {context}\n\n"
            "Important: If the context contains the name of a specific product relevant to the question (e.g., a shampoo for washing dogs), "
            "include the product name in the response and explain how it is useful, citing the source context where it is mentioned. "
            "This helps provide practical recommendations, such as 'TropiClean OxyMed Shampoo is suitable for washing dogs' [citation:2].\n\n"
            "Remember: Cite contexts by their position number (1 for first context, 2 for second, etc.) and do not blindly repeat the contexts verbatim."
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # 修改 create_stuff_documents_chain 来自定义 context 格式
        document_prompt = PromptTemplate.from_template("\n\n- {page_content}\n\n")

        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
            document_variable_name="context",
            document_prompt=document_prompt
        )

        # Create retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )

        # Generate response
        chat_history = []
        for message in messages["messages"]:
            if message["role"] == "user":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                # if include __LLM_RESPONSE__, only use the last part
                if "__LLM_RESPONSE__" in message["content"]:
                    message["content"] = message["content"].split("__LLM_RESPONSE__")[-1]
                chat_history.append(AIMessage(content=message["content"]))

        full_response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        print("Full response (answer):", full_response["answer"])
        # Update bot message content
        bot_message.content = full_response["answer"] if isinstance(full_response, dict) else full_response
        db.commit()
        if isinstance(full_response, dict):
            formatted_response = re.sub(r'\[citation:(\d+)\]|\[\d+\]', '', full_response["answer"])
            formatted_response = (formatted_response
                .replace('\n', ' ')
                .replace('  ', ' ')
                .replace(' .', '. ')
                .replace(' ,', ', ')
                .replace('*', ''))
            return formatted_response
        else: 
            return full_response
            
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        
        # Update bot message with error
        if 'bot_message' in locals():
            bot_message.content = error_message
            db.commit()
        return error_message
    finally:
        db.close()