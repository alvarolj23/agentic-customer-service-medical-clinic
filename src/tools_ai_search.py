from pydantic import BaseModel, Field
from langchain.agents import tool
# Reuse the constant vector store setup
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
import os

import dotenv
dotenv.load_dotenv()

# Option 2: use an Azure OpenAI account with a deployment of an embedding model
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")


index_name: str = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

#os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = "<YOUR_SEARCH_SERVICE_NAME>"
os.environ["AZURE_AI_SEARCH_INDEX_NAME"] = index_name
#os.environ["AZURE_AI_SEARCH_API_KEY"] = "<YOUR_API_KEY>"

azure_ai_search_service_name = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
vector_store_password = os.getenv("AZURE_AI_SEARCH_API_KEY")

# Option 2: Use AzureOpenAIEmbeddings with an Azure account
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
)

# Specify additional properties for the Azure client such as the following https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core/README.md#configurations
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=azure_ai_search_service_name,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    # Configure max retries for the Azure client
    additional_search_client_options={"retry_total": 4},
)

# Define input schema
class ProductSearchInput(BaseModel):
    prod_id: str = Field(..., description="Product ID to search for")
    country_code: str = Field(..., description="Country code to filter the results, e.g., 'DE', 'FR'")
    query: str = Field(..., description="Search query or question to ask about the product")


def search_product_details(prod_id: str, country_code: str, query: str) -> list[dict]:
    """
    Search for product details by Product ID, filter by country, and include relevant document details.
    """
    # Define the filter to locate the product by its ID
    product_filter = f"prod_id eq '{prod_id}'"

    # Perform the search
    search_results = vector_store.similarity_search(
        query=query,
        k=10,  # Number of results to return
        search_type="hybrid",
        filters=product_filter,  # Apply the product ID filter
    )

    matched_details = []
    for result in search_results:
        product_detail = {
            "content": result.metadata.get("content"),
            "filename": result.metadata.get("filename"),
            "prod_id": result.metadata.get("prod_id"),
        }
        # Avoid duplicates
        if product_detail not in matched_details:
            matched_details.append(product_detail)

    return matched_details