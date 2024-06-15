
from dotenv import load_dotenv
from modules.loader import Loader
from modules.models import Router, DocGrader, Generator, HallucinationGrader,AnswerGrader,QuestionRewriter
from modules.nodes  import Nodes
from modules.graph import Graph

load_dotenv()




loader = Loader()


class NodeHelpers:
    retriever = loader.get_retriever()

    question_router = Router.get_model()

    retrieval_grader = DocGrader.get_model()

    rag_chain = Generator.get_model()

    hallucination_grader = HallucinationGrader.get_model()

    answer_grader = AnswerGrader.get_model()

    question_rewriter = QuestionRewriter.get_model()



cite_mapper = {"data\\L1-Introduction.pdf":"https://stanford-cs324.github.io/winter2022/lectures/introduction/",
                "data\\L2-CAPABILITIES.pdf" : "https://stanford-cs324.github.io/winter2022/lectures/capabilities/",
                "data\\L3-Harms.pdf" :"https://stanford-cs324.github.io/winter2022/lectures/harms-1/" ,
                "data\\L4-Harms-II.pdf": "https://stanford-cs324.github.io/winter2022/lectures/harms-2/"}



nodes = Nodes(NodeHelpers())

app = Graph.create(nodes)
# from print import print

# Run
inp = ""

message_history = ["the messages till now are given below: \n\n"]
while inp!="exit":
    inp = input("Enter question: ")
    inputs = {"question": inp,"message_history" : message_history }
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            print(f"Node '{key}': \nREGENERATED : {nodes.REGENERATION_COUNT}\nRETREVIAL : {nodes.RERETREVIAL_COUNT}" )
           
        print("\n---\n")

    # Final generation
    print(value["generation"])
    
    # print(sentence)
    value["message_history"].append("msg number ["+str(len(value["message_history"]))+"] agent : "+value["generation"])
    message_history = value["message_history"]
    # print(message_history)
    for doc in value["documents"]:
        print("site : " + cite_mapper[doc.metadata["source"]] + "\t  page : " +str( doc.metadata["page"]) )

