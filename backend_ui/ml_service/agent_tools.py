from functools import wraps

from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from langchain.agents import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.agents import Tool

from ml_service.webscrapper_codigos_chile.webscrapper_app import qdrant_retriever as ccr
from ml_service.qdrant_vectorstore.vectorstore_funcs import (
    obtain_vectorstore_from_client,
    obtain_llm_multiquery_retriever,
    qdrant_retriever,
)

from utils.config import SERAPI_API_KEY, OPENAI_API_KEY
from ml_service.prompts import rephrased_retriever_template





# PDF TOOL RETRIEVER
# Retriever params
def policy_feature_tool():
    search_type = "mmr"
    search_kwargs = {"k": 3, "lambda_mult": 0.25}

    embedding_name = "openai_embeddings"
    collection_name = "pdf_pol_feature_openai_embeddings"

    pdf_retriever = qdrant_retriever(
        embedding_name=embedding_name,
        collection_name=collection_name,
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    pol_tool = create_retriever_tool(
        retriever=pdf_retriever,
        description="Esquema. Estructura de polizas de seguros. Lista de los articulos.",
        name="policy_feature",
    )

    return pol_tool


def article_feature_tool():
    search_type = "mmr"
    search_kwargs = {"k": 4, "lambda_mult": 0.2}

    embedding_name = "openai_embeddings"
    collection_name = "pdf_art_feature_openai_embeddings"

    pdf_retriever = qdrant_retriever(
        embedding_name=embedding_name,
        collection_name=collection_name,
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    art_tool = create_retriever_tool(
        retriever=pdf_retriever,
        description="Encabezado de los articulos.",
        name="article_feature",
    )

    return art_tool

# PDR CONTENT TOOL
@tool
def content_feature(user_input: str):
    """Recibe la query del user y desarrollar explicaciones para un articulo en particular.
    Funcion para hacer un resumen, buscar detalles de articulos de polizas."""
    
    search_type = "similarity"
    search_kwargs = {"k": 1}

    embedding_name = "openai_embeddings"
    collection_name = "pdf_final_feature_openai_embeddings"

    pdf_retriever = qdrant_retriever(
        embedding_name=embedding_name,
        collection_name=collection_name,
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    
    REPHRASED_RETRIEVER_PROMPT = PromptTemplate.from_template(rephrased_retriever_template)
    
    _llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY)
    
    standalone_query = {
            "input": lambda x: x
        } | REPHRASED_RETRIEVER_PROMPT
    
    
    
    # import pickle, time
    # res = standalone_query.invoke(user_input)
    # save = (user_input, res)
    # with open(f'save_{int(time.time())}.p', 'wb') as f:
    #     pickle.dump(save, f)

    rephrase_chain = standalone_query | StrOutputParser() | pdf_retriever    
    try:
        return rephrase_chain.invoke(user_input)[0].page_content
    except Exception as e:
        return f"Calling tool with arguments:\n\n{user_input}\n\nraised the following error:\n\n{type(e)}: {e}"
    

# SERAPI TOOL
def to_string(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return str(func(*args, **kwargs))

    return wrapper


def web_news_tool():
    params = {
        "engine": "google",
        "gl": "cl",
        "hl": "es",
        "q": "seguros de vida",
        "tbm": "nws",
    }
    search = SerpAPIWrapper(
        params=params,
        serpapi_api_key=SERAPI_API_KEY,
    )

    # Create the function for the tool
    @to_string
    def dfunc(*args, **kwargs):
        return search.run(*args, **kwargs)

    news_tool = Tool(
        name="web_news",
        description="Noticias y novedades en internet y la web sobre empresas de seguros.",
        func=dfunc,
    )

    return news_tool


def retriever_tool_constitucion_chile():
    # CONSTITUCION CHILE TOOL
    # web scrapper params
    searchs = [
        ("codigo_de_comercio", [524, 525, 526]),  # , 538]),
        ("companias_de_seguros", [3, 10, 36]),
        ("protocolo_seguridad_sanitaria", [18]),
        ("codigo_sanitario", [112]),
        ("codigo_penal", [470]),
    ]

    # Obtain the retrievers
    search_type = "mmr"
    search_kwargs = {"k": 3, "lambda_mult": 0.25}
    web_retriever = ccr(searchs, search_type, search_kwargs)

    web_tool = create_retriever_tool(
        retriever=web_retriever,
        description="Constitucion de Chile. Codigo de comercio, Compañias de seguros, Protocolo de seguridad sanitaria, Codigo sanitario, Codigo penal.",
        name="cl_constit_tool",
    )
    return web_tool
