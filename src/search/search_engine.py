from opensearchpy import OpenSearch
from src.config.config import OPENSEARCH_CONFIG, INDEX_NAME, SEARCH_TOP_K, RERANK_TOP_K, FINAL_TOP_K, IMPORTANT_KEYWORDS
from src.models.embedding_models import model, reranker
from src.utils.text_processing import preprocess_text
import re

# Initialize OpenSearch client
client = OpenSearch(**OPENSEARCH_CONFIG)

def keyword_relevance_score(query, doc):
    """Tính điểm relevance dựa trên từ khóa"""
    query_lower = query.lower()
    doc_lower = doc.lower()
    
    score = 0
    
    # Tìm tên người
    names = re.findall(r'\b[A-ZÀÁẢÃẠÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƯỪỨỬỮỰĐ][a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ]+(?:\s+[A-ZÀÁẢÃẠÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƯỪỨỬỮỰĐ][a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ]+)*', query)
    for name in names:
        if name.lower() in doc_lower:
            score += 3
    
    # Tìm năm sinh
    years = re.findall(r'\b(19|20)\d{2}\b', query)
    for year in years:
        if year in doc_lower:
            score += 2
    
    # Từ khóa liên quan đến sinh năm
    if any(word in query_lower for word in ["sinh", "năm", "born"]):
        if any(word in doc_lower for word in ["sinh", "năm", "born"]):
            score += 1
    
    # Các từ khóa quan trọng khác
    for keyword in IMPORTANT_KEYWORDS:
        if keyword in query_lower and keyword in doc_lower:
            score += 1
    
    return score

def search_similar(query_text, top_k=SEARCH_TOP_K):
    """Tìm kiếm kết hợp vector similarity và text search"""
    
    # Tiền xử lý query
    processed_query = preprocess_text(query_text)
    
    # Tạo embedding cho query
    query_embedding = model.encode([processed_query])[0].tolist()
    
    # Tìm kiếm kết hợp KNN và text search
    search_query = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": top_k
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": processed_query,
                                "boost": 0.5
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "text": {
                                "query": processed_query,
                                "boost": 1.0
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "_source": ["text", "chunk_index", "length"]
    }
    
    try:
        response = client.search(index=INDEX_NAME, body=search_query)
        hits = response["hits"]["hits"]
        
        # Lấy text và tiền xử lý
        docs_with_score = []
        for hit in hits:
            text = preprocess_text(hit["_source"]["text"])
            score = hit["_score"]
            docs_with_score.append((text, score))
        
        # Sắp xếp theo score
        docs_with_score.sort(key=lambda x: x[1], reverse=True)
        docs = [doc for doc, score in docs_with_score]
        
        print(f"🔍 Tìm thấy {len(docs)} documents")
        return docs
        
    except Exception as e:
        print(f"❌ Lỗi search: {e}")
        # Fallback về KNN search đơn giản
        return search_knn_only(query_text, top_k)

def search_knn_only(query_text, top_k=SEARCH_TOP_K):
    """Fallback: chỉ sử dụng KNN search"""
    processed_query = preprocess_text(query_text)
    query_embedding = model.encode([processed_query])[0].tolist()
    
    search_query = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        },
        "_source": ["text"]
    }
    
    try:
        response = client.search(index=INDEX_NAME, body=search_query)
        hits = response["hits"]["hits"]
        docs = [preprocess_text(hit["_source"]["text"]) for hit in hits]
        print(f"🔍 KNN Fallback: Tìm thấy {len(docs)} documents")
        return docs
    except Exception as e:
        print(f"❌ Lỗi KNN search: {e}")
        return []

def rerank_docs(query, docs, top_k=RERANK_TOP_K):
    """Cải thiện reranking với kết hợp keyword matching và cross-encoder"""
    
    if not docs:
        return []
    
    # Bước 1: Lọc theo keyword relevance
    keyword_scores = [(doc, keyword_relevance_score(query, doc)) for doc in docs]
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy top docs có keyword relevance cao
    high_keyword_docs = [doc for doc, score in keyword_scores[:15] if score > 0]
    remaining_docs = [doc for doc, score in keyword_scores if score == 0][:10]
    
    # Kết hợp
    docs_to_rerank = high_keyword_docs + remaining_docs
    
    # Bước 2: Sử dụng cross-encoder cho reranking
    if len(docs_to_rerank) == 0:
        docs_to_rerank = docs[:15]
    
    try:
        pairs = [[query, doc] for doc in docs_to_rerank]
        cross_scores = reranker.predict(pairs)
        
        # Kết hợp điểm keyword và cross-encoder
        final_scores = []
        for i, (doc, cross_score) in enumerate(zip(docs_to_rerank, cross_scores)):
            keyword_score = keyword_relevance_score(query, doc)
            # Trọng số: 70% cross-encoder, 30% keyword
            final_score = 0.7 * cross_score + 0.3 * keyword_score
            final_scores.append((doc, final_score, keyword_score, cross_score))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Debug info
        print(f"📊 Reranking scores:")
        for i, (doc, final_score, kw_score, cross_score) in enumerate(final_scores[:5], 1):
            print(f"  {i}. Final: {final_score:.3f} (KW: {kw_score}, Cross: {cross_score:.3f}) - {doc[:80]}...")
        
        return [doc for doc, _, _, _ in final_scores[:top_k]]
    
    except Exception as e:
        print(f"❌ Reranker error: {e}")
        # Fallback về keyword ranking
        return [doc for doc, score in keyword_scores[:top_k]] 