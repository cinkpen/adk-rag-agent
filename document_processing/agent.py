from google.adk.agents import Agent
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent  # Add this import if ParallelAgent is provided by the library
from vertexai.preview.generative_models import GenerativeModel, Tool
from vertexai.preview import rag
from google.cloud import storage
import os
import fitz  # PyMuPDF for PDF processing
import io
from PIL import Image

RAG_CORPUS_NAME = "projects/agentic-demo-002-chris/locations/us-central1/ragCorpora/{{corpusId}}"

domains = [
        {"domainId": "wiringdiagram", "projectId": "abmproject1", "name": "Wiring Diagram", "corpusURI": "projects/agentic-demo-002-chris/locations/us-central1/ragCorpora/1152921504606846976"},
        {"domainId": "vavschedule", "projectId": "abmproject1", "name": "VAV Schedule", "corpusURI": ""}
    ]

def getRagCorpusURI(projectId: str, domainId: str) -> str:
    """
    Returns the RAG corpus name based on the projectId and domainId.
    """

    corpusId = [domain for domain in domains if domain["domainId"] == domainId and domain["projectId"] == projectId][0].get("corpusURI")

    return corpusId if corpusId else ""

def extract_images_from_pdf(project_id, domain_id, pdf_file):
    """
    Extracts images from a PDF file passed via ADK web as an attachment.
    
    Args:
        project_id (str): The project ID
        domain_id (str): The domain ID
        pdf_file: The PDF file object from ADK web attachment
    
    Returns:
        list: A list of extracted image paths/URIs
    """
    extracted_images = []
    
    try:
        # Read the PDF file content
        pdf_content = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer for potential reuse
        
        # Open PDF document from bytes
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Initialize Google Cloud Storage client
        storage_client = storage.Client()
        
        # Create the bucket if it doesn't exist
        bucket = storage_client.bucket(project_id)
        if not bucket.exists():
            bucket.location = "us-central1"  # Set your preferred location
            storage_client.create_bucket(bucket)
        
        # Extract images from each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                # Get the XREF of the image
                xref = img[0]
                
                # Extract the image
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Create a filename for the extracted image
                image_filename = f"{project_id}-{domain_id}-page{page_num + 1}-img{img_index + 1}.{image_ext}"
                
                try:
                    # Upload image to Google Cloud Storage
                    blob = storage_client.bucket(project_id).blob(f"extracted_images/{image_filename}")
                    blob.upload_from_string(image_bytes, content_type=f"image/{image_ext}")
                    
                    image_gcs_uri = f"gs://{project_id}/extracted_images/{image_filename}"
                    extracted_images.append(image_gcs_uri)
                    
                    print(f"Extracted and uploaded image: {image_gcs_uri}")
                    
                except Exception as e:
                    print(f"Failed to upload extracted image {image_filename}: {e}")
        
        pdf_document.close()
        
        # Also upload the original PDF file
        try:
            pdf_filename = f"{project_id}-{domain_id}-{pdf_file.filename}"
            blob = storage_client.bucket(project_id).blob(pdf_filename)
            blob.upload_from_file(pdf_file, content_type=pdf_file.content_type)
            uploaded_pdf_uri = f"gs://{project_id}/{pdf_filename}"
            
            # Notify RAG engine about the new PDF document
            corpus_uri = getRagCorpusURI(project_id, domain_id)
            if corpus_uri:
                try:
                    rag.import_files(
                        corpus_uri,
                        paths=[uploaded_pdf_uri], 
                        # Optional: Add transformation_config for chunking during import
                        # transformation_config=rag.TransformationConfig(
                        #     chunking_config=rag.ChunkingConfig(
                        #         chunk_size=512, # Example chunk size
                        #         chunk_overlap=100, # Example chunk overlap
                        #     ),
                        # ),
                    )
                    print(f"Triggered RAG ingestion for PDF document: {uploaded_pdf_uri} into corpus: {corpus_uri}")
                except Exception as e:
                    print(f"Failed to trigger RAG ingestion for PDF document: {uploaded_pdf_uri} into corpus: {corpus_uri}. Error: {e}")
            else:
                print(f"No RAG corpus URI configured for project {project_id} and domain {domain_id}. Skipping RAG ingestion.")
                
        except Exception as e:
            print(f"Failed to upload original PDF document: {e}")
        
        print(f"Successfully extracted {len(extracted_images)} images from PDF")
        return extracted_images
        
    except Exception as e:
        print(f"Failed to extract images from PDF: {e}")
        return []

def fine_tuned_model_process(image_path: str) -> str:
    """
    A tool that processes images for fine-tuning a model.

    Args:
        image_path (str): The path to the image to be processed.
    Returns:
        str: A confirmation message indicating the image has been processed.
    """
    # Placeholder for actual image processing logic
    print(f"Processing image for fine-tuning: {image_path}")
    # Simulate image processing
    return f"Image {image_path} processed for fine-tuning."

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


image_onboarder_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="image_onboarder_agent",
    description="An agent that sends images to be processed for the fine tuning process before adding to RAG Corpus.",
    instruction="""You are a BMS expert. Your role is to send extracted images from PDF documents to be processed 
    for the fine-tuning process before adding them to the RAG Corpus. You must send the images using the tool called "fine_tuned_model_process".
    If you cannot send an extracted image, please say so.""",
    tools=[fine_tuned_model_process]
)

pdf_onboarder_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="pdf_onboarder_agent",
    description="An agent that extracts images from PDF documents.",
    instruction="""You are a BMS expert. Your role is to extract images from PDF documents. 
    If you cannot extract an image, please say so.""",
    tools=[extract_images_from_pdf], 
    output_key="pdf_onboarder_agent_response",
    sub_agents=[image_onboarder_agent],
)

parallel_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[pdf_onboarder_agent]
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