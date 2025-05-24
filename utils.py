import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import json
import os
import pickle
import subprocess
import requests
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load .env file
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def preprocess_kcc(input_path='KCC-DataSet.csv', output_path='processed_docs.json'):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['QueryText', 'KccAns'])
    docs = []
    for idx, row in df.iterrows():
        text = (
            f"Query: {row['QueryText']}\nAnswer: {row['KccAns']}\n"
            f"Meta: State={row['StateName']}, District={row['DistrictName']}, Crop={row['Crop']}"
        )
        docs.append({"id": str(idx), "content": text})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(docs, f, indent=2)

def build_vector_store(doc_path='processed_docs.json', index_path='faiss_index.pkl'):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    with open(doc_path) as f:
        docs = json.load(f)
    documents = [Document(page_content=d['content']) for d in docs]
    db = FAISS.from_documents(documents, embed_model)
    with open(index_path, 'wb') as f:
        pickle.dump(db, f)

def retrieve_context_with_score(query, index_path='faiss_index.pkl', top_k=3, score_threshold=1.0):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(index_path):
        return None, 0
    with open(index_path, 'rb') as f:
        db = pickle.load(f)
    results = db.similarity_search_with_score(query, k=top_k)
    if not results:
        return None, 0
    best_distance = results[0][1]
    if best_distance > score_threshold:
        return None, best_distance
    context = "\n\n".join([doc.page_content for doc, dist in results])
    return context, best_distance

def ask_gemma(query, context):
    prompt = f"""Answer the user's question using the following agricultural advice:\n\n{context}\n\nQuestion: {query}"""
    result = subprocess.run(
        ["ollama", "run", "gemma:2b", prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout.strip()

def fallback_live_search(query, api_key, num_results=5):
    """
    Perform live internet search using SerpAPI when no local context is found.
    This function searches for agricultural information related to the user's query.
    """
    if not api_key:
        return "Live search unavailable: No API key provided."
    
    # Enhance query with agricultural context for better results
    enhanced_query = f"agricultural advice {query} farming tips"
    
    # SerpAPI endpoint for Google search
    search_url = "https://serpapi.com/search"
    
    params = {
        "engine": "google",
        "q": enhanced_query,
        "api_key": api_key,
        "num": num_results,
        "hl": "en",
        "gl": "us"
    }
    
    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        data = response.json()
        
        # Extract relevant information
        results = []
        
        # Check for organic results
        if "organic_results" in data:
            for result in data["organic_results"]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                
                if snippet:  # Only include results with snippets
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "source": link
                    })
        
        # Check for featured snippet (answer box)
        if "answer_box" in data:
            answer_box = data["answer_box"]
            if "snippet" in answer_box:
                results.insert(0, {
                    "title": "Featured Answer",
                    "snippet": answer_box["snippet"],
                    "source": answer_box.get("link", "")
                })
        
        # Format results for display
        if results:
            formatted_results = []
            for i, result in enumerate(results[:3], 1):  # Limit to top 3 results
                formatted_result = f"**Result {i}:** {result['title']}\n"
                formatted_result += f"{result['snippet']}\n"
                formatted_result += f"*Source: {result['source']}*"
                formatted_results.append(formatted_result)
            
            return "\n\n---\n\n".join(formatted_results)
        else:
            return "No relevant search results found for your agricultural query."
            
    except requests.exceptions.RequestException as e:
        return f"Live search unavailable: Network error ({str(e)})"
    except KeyError as e:
        return f"Live search unavailable: Invalid API response ({str(e)})"
    except Exception as e:
        return f"Live search unavailable: {str(e)}"

def fallback_search_alternative(query, num_results=3):
    """
    Alternative fallback using DuckDuckGo search (no API key required)
    This can be used as a backup when SerpAPI is not available.
    """
    try:
        # You would need to install duckduckgo-search: pip install duckduckgo-search
        from duckduckgo_search import DDGS
        
        enhanced_query = f"agricultural advice {query} farming"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(enhanced_query, max_results=num_results))
            
        if results:
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_result = f"**Result {i}:** {result['title']}\n"
                formatted_result += f"{result['body']}\n"
                formatted_result += f"*Source: {result['href']}*"
                formatted_results.append(formatted_result)
            
            return "\n\n---\n\n".join(formatted_results)
        else:
            return "No relevant search results found."
            
    except ImportError:
        return "Alternative search unavailable: duckduckgo-search not installed."
    except Exception as e:
        return f"Alternative search unavailable: {str(e)}"

# Enhanced version that tries multiple search methods
def comprehensive_fallback_search(query, api_key=None):
    """
    Comprehensive fallback that tries SerpAPI first, then DuckDuckGo as backup.
    """
    # Try SerpAPI first if API key is available
    if api_key:
        serpapi_result = fallback_live_search(query, api_key)
        if not serpapi_result.startswith("Live search unavailable"):
            return f"🔍 **Live Internet Search Results:**\n\n{serpapi_result}"
    
    # Fallback to DuckDuckGo search
    ddg_result = fallback_search_alternative(query)
    if not ddg_result.startswith("Alternative search unavailable"):
        return f"🔍 **Live Internet Search Results (via DuckDuckGo):**\n\n{ddg_result}"
    
    # If both fail, return a helpful message
    return ("🚫 **Live search currently unavailable.**\n\n"
            "Please check your internet connection or API key configuration. "
            "You can try rephrasing your query or ask about topics covered in the local KCC dataset.")