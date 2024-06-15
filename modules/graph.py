from langgraph.graph import END, StateGraph
from typing import List

from typing_extensions import TypedDict




class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    message_history: List[str]



class Graph:
    @staticmethod
    def create(nodes):
        # self.nodes = nodes
        workflow = StateGraph(GraphState)

        # Define the nodes
        # workflow.add_node("web_search", web_search)  # web search
        workflow.add_node("retrieve", nodes.retrieve)  # retrieve
        workflow.add_node("grade_documents", nodes.grade_documents)  # grade documents
        workflow.add_node("generate", nodes.generate)  # generatae
        workflow.add_node("generate_conv_reply", nodes.generate)
        workflow.add_node("transform_query", nodes.transform_query)  # transform_query

        # Build graph
        workflow.set_conditional_entry_point(
            nodes.route_question,
            {
                "casual_convo": "generate_conv_reply",
                "vectorstore": "transform_query",
            },
        )
        # workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("generate_conv_reply", END)
        workflow.add_conditional_edges(
            "grade_documents",
            nodes.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            nodes.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        return workflow.compile()