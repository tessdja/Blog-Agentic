from src.states.blogstate import BlogState

class BlogNode:
    """ 
    A class to representthe blog node
    """

    def __init__(self,llm):
        self.llm=llm

    def title_creation(self,state:BlogState):
        """ 
        create the title for the blog
        """

        if "topic" in state and state["topic"]:
            prompt=""" 
                    You are an expert blog content writer. USe Markdown formatting. Generate
                    a blog title for the {topic}. This title should be creative and SEO friendly.

                    """
            
            system_message=prompt.format(topic=state["topic"])
            response=self.llm.invoke(system_message)
            return{"blog":{"title":response.content}}
        
    def content_generation(self, state:BlogState):
        if "topic" in state and state["topic"]:
            system_prompt = """You are an expert blog writer. Use Markdown formatting and
             Generate a detailed blog content with detailed breakdown for the {topic} """
            system_message = system_prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": state['blog']['title'], "content": response.content}}
        
