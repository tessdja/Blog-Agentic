from langgraph.graph import StateGraph, START, END
from src.llms.groqllm import GroqLLM
from src.states.blogstate import BlogState
from src.nodes.blog_node import BlogNode

class GraphBuilder:
    def __init__(self, llm):
        self.llm=llm
        self.graph=StateGraph(BlogState)

    def build_topic_graph(self):
        """ 
        Build a graph to generate blogs based on topic
        """

        self.blog_node_obj=BlogNode(self.llm)
        ## Nodes
        self.graph.add_node("title creation", self.blog_node_obj.title_creation)
        self.graph.add_node("content_generation", self.blog_node_obj.content_generation)

        ## Edges
        self.graph.add_edge(START, "title creation")
        self.graph.add_edge("title creation", "content_generation")
        self.graph.add_edge("content_generation", END) 

        return self.graph
    
    def setup_graph(self,usecase):
        if usecase=="topic":
            self.build_topic_graph()

        return self.graph.compile()
