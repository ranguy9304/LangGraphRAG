from typing import List

from typing_extensions import TypedDict

from langchain.schema import Document
# if debug_mode:from 
#   if debug_mode:print i mport 
# print
from dotenv import load_dotenv
import os
load_dotenv()
debug_mode = os.getenv("DEBUG") == "True"


class Nodes:
    REGENERATION_THRESHOLD = int(os.getenv("REGENERATION_THRESHOLD"))
    RERETREVIAL_THRESHOLD = int(os.getenv("RERETREVIAL_THRESHOLD"))
    REGENERATION_COUNT = 0
    RERETREVIAL_COUNT = 0
    
    def __init__(self,helpers):
        self.helpers = helpers
        pass
    def retrieve(self,state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        # self.REGENERATION_COUNT =0
        if debug_mode:
            print("---RETRIEVE---")
        question = state["question"]
        message_history = state["message_history"]
        # Retrieval
        documents = self.helpers.retriever.invoke(question)
        return {"documents": documents, "question": question,"message_history":message_history}


    def generate(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        if debug_mode:
            print("---GENERATE---")
        question = state["question"]
        if state["documents"]:
            documents = state["documents"]
        else:
            documents = []
        message_history = state["message_history"]

        context_in = documents.copy()
        for messages in message_history:
            context_in.append(messages)

        # RAG generation
        generation = self.helpers.rag_chain.invoke({"context": context_in, "question": question},config={
            "configurable": {"session_id": "abc123"}
        },)
        return {"documents": documents, "question": question, "generation": generation,"message_history":message_history}



    def grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        if debug_mode:
            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        message_history = state["message_history"]
        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.helpers.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                if debug_mode:
                    print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                if debug_mode:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question,"message_history":message_history}


    def transform_query(self,state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        if debug_mode:
            print("---TRANSFORM QUERY---")
        question = state["question"]
        if state["documents"]:
            documents = state["documents"]
        else:
            documents = []
        message_history = state["message_history"]

        
        # Re-write question
        better_question = self.helpers.question_rewriter.invoke({"question": question, "chat_history": message_history})
        if debug_mode:
            print("better question : ",better_question)
        return {"documents": documents, "question": better_question,"message_history":message_history}


    


    ### Edges ###


    def route_question(self,state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        if debug_mode:
            print("---ROUTE QUESTION---")
        question = state["question"]
        state["message_history"].append("msg number ["+str(len(state["message_history"]))+"] user : "+question)

        self.REGENERATION_COUNT = 0
        self.RERETREVIAL_COUNT = 0

        source = self.helpers.question_router.invoke({"question": question})
        if source.datasource == "casual_convo":
            if debug_mode:
                print("---ROUTE QUESTION TO CASUAL CONVO---")
            return "casual_convo"
        elif source.datasource == "vectorstore":
            if debug_mode:
                print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"


    def decide_to_generate(self,state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        if debug_mode:
            print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if (not filtered_documents) and self.RERETREVIAL_COUNT < self.RERETREVIAL_THRESHOLD:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            if debug_mode:
                print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            self.RERETREVIAL_COUNT +=1
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            if debug_mode:
                print("---DECISION: GENERATE---")
            self.REGENERATION_COUNT+=1
            return "generate"


    def grade_generation_v_documents_and_question(self,state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        if debug_mode:
            print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.helpers.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes" or self.REGENERATION_COUNT > self.REGENERATION_THRESHOLD:
            if debug_mode and grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            if debug_mode and self.REGENERATION_COUNT > self.REGENERATION_THRESHOLD:
                print("---DECISION: REGENERATION THRESHOLD REACHED---")
            # Check question-answering
            if debug_mode:
                print("---GRADE GENERATION vs QUESTION---")
            score = self.helpers.answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes" or self.REGENERATION_COUNT > self.REGENERATION_THRESHOLD or self.RERETREVIAL_COUNT > self.RERETREVIAL_THRESHOLD:
                if debug_mode:
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                # self.REGENERATION_COUNT = 0
                self.RERETREVIAL_COUNT+=1
                if debug_mode:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            self.REGENERATION_COUNT+=1
            if debug_mode:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
