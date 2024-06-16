from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


from dotenv import load_dotenv




load_dotenv()


class Router():

    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""

        datasource: Literal["vectorstore", "casual_convo"] = Field(
            ...,
            description="Given a user question choose to route it to casual_convo or a vectorstore.",
        )


    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Prompt
    system = """You are an expert at routing a user question to a vectorstore or casual_convo.
    choose the vector store if the question is related to CS324 and casual_convo if it is not. \n
    The vectorstore contains documents related to CS324! This is a new course on understanding and developing large language models  introduction to large language models (LLMs), 
    their capabilities, various harms including performance disparities, social biases, toxicity, disinformation, and mitigation strategies, along with relevant case studies and high-level ideas from other disciplines..
    Use the vectorstore for questions on these topics which require some data and follow up questions. Otherwise if only normal response and chat history is required , use casual_convo.
    
    for example:
        user: Hi what are astonaughts [ this is a random question not related to CS324 so route to casual_convo] : casual_convo
        user: What are the harms of llms? [ this is related to CS324 so route to vectorstore] : vectorstore
        """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
    @staticmethod
    def get_model():
        return Router.question_router
# print(
#     question_router.invoke(
#         {"question": "Who will the Bears draft first in the NFL draft?"}
#     )
# )
# print(question_router.invoke({"question": "What are harms of llms?"}))



class DocGrader():

    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )


    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    @staticmethod
    def get_model():
        return DocGrader.retrieval_grader


class Generator:
    prompt = ChatPromptTemplate.from_messages([
            ("system",  """You are a speacilised assistant for question-answering tasks only related to given topic. Use the following pieces of retrieved context and message history to answer the question.
             If the answer is not provided in the retrieved documents and message history, just say that you don't know. Keep the answer detailed. If question is not related to LLM (Large language models)
             and other ai topics just reply with "lets keep the talk relevant to llms"."""),
            ("human", """Use the following pieces of retrieved context and message history to answer the question.
             If the answer is not provided in the retrieved documents and message history, just say that you don't know only use the context to answer the question. Keep the answer detailed. If question is not related to LLM (Large language models)
             and other ai topics just reply with "lets keep the talk relevant to llms".
                Question: {question}        
                Context: {context} 
                Answer:"""),
                ])

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    @staticmethod
    def get_model():
        return Generator.rag_chain


class HallucinationGrader:
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )


    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts. 
        If the LLM Generation is saying that it doesnt know or not sure or stating to keep the questions relevant to topic , grade it as 'yes'."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "If the LLM Generation is saying that it doesnt know or not sure or stating to keep the questions relevant to topic , grade it as 'yes'. Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    @staticmethod
    def get_model():
        return HallucinationGrader.hallucination_grader


class AnswerGrader:
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )


    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
        If the LLM Generation is saying that it doesnt know or not sure or stating to keep the questions relevant to topic , grade it as 'yes'."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "If the LLM Generation is saying that it doesnt know or not sure or stating to keep the questions relevant to topic , grade it as 'yes'. User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    @staticmethod
    def get_model():
        return AnswerGrader.answer_grader

class QuestionRewriter:
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval, and very concise. Look at the input and try to reason about the underlying semantic intent / meaning. The input can also be a
        follow up question, look at the chat history to re-write the question to include necessary info from the chat history to a better version that is optimized \n 
        for vectorstore retrieval without any other info needed. [the topic of convo will be generally around  introduction to large language models (LLMs),
         their capabilities, various harms including performance disparities, social biases, toxicity, disinformation, and mitigation strategies, along with relevant case studies and high-level ideas from other disciplines.]"""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question \n\n CHAT HISTORY : {chat_history}.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    @staticmethod
    def get_model():
        return QuestionRewriter.question_rewriter
