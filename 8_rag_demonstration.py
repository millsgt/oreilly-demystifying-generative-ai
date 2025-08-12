
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from MyKey import MY_OPENAI_KEY

# Initialize the OpenAI client
client = OpenAI(api_key=MY_OPENAI_KEY)


# Sample proprietary company documents (dummy data representing internal information not typically known by an LLM)
documents = [
    "TechCorp Internal Memo: The CEO, Jane Doe, announced a new AI initiative called Project Nebula, aimed at revolutionizing cloud computing.",
    "TechCorp Employee Handbook: All employees must complete annual cybersecurity training by December 31st. Failure to comply results in restricted access to company systems.",
    "TechCorp Financial Report Q2 2024: Revenue reached $150 million, with a 20% increase in sales from the proprietary Quantum Processor line. Net profit margin is 15%.",
    "TechCorp R&D Update: The proprietary algorithm for predictive analytics, codenamed 'Oracle Engine', has achieved 95% accuracy in beta testing and is patented under US Patent #1234567.",
    "TechCorp HR Policy: Remote work is allowed for up to 3 days per week, but requires approval from department heads. Proprietary information must not be shared outside secure channels.",
    "TechCorp Product Roadmap: Version 2.0 of the Nebula Platform includes enhanced encryption features to protect sensitive client data, scheduled for release in Q1 2025.",
]


# Function to get embeddings for a list of texts
def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Use an embedding model
        input=texts
    )
    return [embedding.embedding for embedding in response.data]


# Precompute embeddings for the documents
document_embeddings = np.array(get_embeddings(documents))


# Function to retrieve relevant documents based on query
def retrieve_documents(query, top_k=2):
    query_embedding = np.array(get_embeddings([query])[0]).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]


# Function to generate response using RAG
def rag_query(query):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)
    context = "\n".join(retrieved_docs)

    # Prepare the prompt with context
    prompt = f"""Use the following context to answer the question. The context contains proprietary company information:
Context:
{context}

Question: {query}
Answer:"""

    # Call the chat completion API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use a suitable model
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to internal company documents."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# Example usage with a query related to the proprietary data
query = "What is Project Nebula at TechCorp?"
answer = rag_query(query)
print(f"Query: {query}")
print(f"Answer: {answer}")