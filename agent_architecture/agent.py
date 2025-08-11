from google.adk.agents import Agent
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent  # Add this import if ParallelAgent is provided by the library
from vertexai.preview.generative_models import GenerativeModel, Tool
from vertexai.preview import rag

RAG_CORPUS_NAME = "projects/agentic-demo-002-chris/locations/us-central1/ragCorpora/{{corpusId}}"

def callRag(corpusId:str, user_query: str) -> str:
    """
    A tool that performs a Retrieval-Augmented Generation (RAG) query using Vertex AI.

    Args:
        corpusId (str): The ID of the RAG corpus to query.
        user_query (str): The query string to search in the RAG corpus.
    Returns:
        string: This function returns the RAG response
    """
    rag_corpus_name = RAG_CORPUS_NAME.replace("{{corpusId}}", corpusId)
    # rag_corpus_name = "projects/agentic-demo-002-chris/locations/us-central1/ragCorpora/3379951520341557248"

    print("RAG Corpus Name:", rag_corpus_name)  # Debugging line to check the RAG corpus name
    try:
        rag_resource = rag.RagResource(rag_corpus=rag_corpus_name)

        rag_retrieval_tool = Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=[rag_resource],
                    similarity_top_k=5,  # Number of top similar documents to retrieve
                    vector_distance_threshold=0.5 # Optional: A threshold for vector distance
                ),
            )
        )

        # 3. Initialize a Generative Model with the RAG Tool
        # You can use models like "gemini-1.5-flash", "gemini-1.0-pro", etc.
        model = GenerativeModel(
            model_name="gemini-2.5-flash", # Or your preferred Gemini model
            tools=[rag_retrieval_tool]
        )

        # 4. Perform a query
        response = model.generate_content(user_query)
       

        print("CorpusId:", corpusId)
        print("Response:", response.text)
        result = response.text
        print("RAG query completed successfully.", result)
        return result
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

graphical_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="graphical_agent",
    description="An agent that provides graphical representations of data in a BRICK schema.",
    instruction="""You are a BMS expert and a BRICK schema expert. One of your roles is to answer the users questions about the 
    BMS system utilising the RAG engine by using the tool called "callRag" as  much as is possible. Your other role is to generate JSON, 
the output should be a JSON array. There should always be a site and any other entities in the 
user query should be added too. 
Each Key must be unique, and each object contain fields: id, name, and type

An example output is:
Model:
[
        {
            "key": "0",
            "data": { "name": "Site1",  "type": "Site" },
            "children": [
                {
                    "key": "4",
                    "data": { "name": "Floor1", "type": "Floor" },
                    "children": [
                        {
                            "key": "4-0",
                            "data": { "name": "Room1","type": "Room" },
                            "children": [
                                {
                                    "key": "4-1-0",
                                    "data": { "name": "AHU1", "type": "Equipment" }
                                },
                                {
                                    "key": "4-1-1",
                                    "data": { "name": "AHU2", "type": "Equipment" }
                                }
                            ]
                        },
                        {
                            "key": "4-1",
                            "data": { "name": "Room2",  "type": "Room" }
                        }
                    ]
                }
            ]
        },
        {
            "key": "5",
            "data": { "name": "Site2", "type": "Folder" },
            "children": []
        }
    ]""",
    tools=[callRag],
        output_key="graphical_response"  # Ensure the output key is set for the response

)

textual_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="textual_agent",
    description="An agent that provides textual responses to users questions.",
    instruction="""You are a BMS expert. Your role is to answer the users questions about the BMS system utilising the RAG engine 
    by using the tool called "callRag" as  much as is possible. If you cannot answer the question, please say so.
    If you can answer the question, please do so in a concise manner.""",
    tools=[callRag],
    output_key="textual_response"  # Ensure the output key is set for the response
)

parallel_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[graphical_agent, textual_agent]
)

synthesiser_agent = LlmAgent(
    model="gemini-2.5-flash",   

    name="synthesiser_agent",
    description="An agent that synthesises the results of graphical_agent and textual_agent.",
    instruction="""You are a BMS expert. Your role is to synthesise the results of graphical_agent and textual_agent.
    You should return a JSON object with two keys: "graphical_response" and "textual_response".
    The "graphical_response" should be the JSON representation from graphical_agent, and the "textual_response" should be the concise text response from textual_agent.
    If either agent fails to provide a response, you should indicate that in the respective key.""")

root_agent = SequentialAgent(
     name="ParallelWebResearchAgent",
     sub_agents=[parallel_agent, synthesiser_agent],
     description="""Runs textual_agent  and graphical_agent to extract different views on the same topic. The result of graphical_agent will be a JSON representation, while the result of textual_agent will be a concise text response.      Both should be returned in the final response as a json structure with keys "graphical_response" and "textual_response" respectively.     """,
     
)