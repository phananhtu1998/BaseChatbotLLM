import re
from typing import List, Dict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import SearchResult, RankedResult

class ContentReranker:
    """Reranker để sắp xếp lại kết quả search theo độ liên quan."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.stop_words_vi = {
            'và', 'của', 'có', 'được', 'này', 'đó', 'một', 'là', 'trong', 
            'với', 'các', 'không', 'cho', 'từ', 'về', 'như', 'để', 'khi',
            'hay', 'hoặc', 'nhưng', 'nếu', 'vì', 'do', 'bởi', 'theo'
        }
    
    def rerank_hybrid(self, query: str, search_results: List[SearchResult], 
                     top_k: int = 3, weights: Dict[str, float] = None) -> List[RankedResult]:
        """Hybrid reranking kết hợp nhiều phương pháp."""
        if weights is None:
            weights = {
                'relevance': 0.4,
                'quality': 0.3,
                'freshness': 0.1,
                'authority': 0.2
            }
        
        if not search_results:
            return []
        
        # Calculate different scores
        relevance_scores = self._calculate_relevance_scores(query, search_results)
        quality_scores = [self._calculate_quality_score(result) for result in search_results]
        freshness_scores = [self._calculate_freshness_score(result) for result in search_results]
        authority_scores = [self._calculate_authority_score(result) for result in search_results]
        
        # Normalize scores
        relevance_scores = self._normalize_scores(relevance_scores)
        quality_scores = self._normalize_scores(quality_scores)
        freshness_scores = self._normalize_scores(freshness_scores)
        authority_scores = self._normalize_scores(authority_scores)
        
        # Combine scores
        ranked_results = []
        for i, result in enumerate(search_results):
            combined_score = (
                relevance_scores[i] * weights['relevance'] +
                quality_scores[i] * weights['quality'] +
                freshness_scores[i] * weights['freshness'] +
                authority_scores[i] * weights['authority']
            )
            
            ranked_results.append(RankedResult(
                original_result=result,
                relevance_score=relevance_scores[i],
                quality_score=quality_scores[i],
                combined_score=combined_score,
                rank_position=i + 1
            ))
        
        # Sort and update positions
        ranked_results.sort(key=lambda x: x.combined_score, reverse=True)
        for i, result in enumerate(ranked_results):
            result.rank_position = i + 1
            
        return ranked_results[:top_k]
    
    def _calculate_relevance_scores(self, query: str, search_results: List[SearchResult]) -> List[float]:
        query_clean = self._clean_text(query)
        documents = [self._clean_text(f"{result.title} {result.description} {result.content}") 
                    for result in search_results]
        
        tfidf_scores = self._calculate_tfidf_similarity(query_clean, documents)
        
        query_keywords = set(self._extract_keywords(query))
        keyword_scores = []
        
        for result in search_results:
            result_text = f"{result.title} {result.description} {result.content}"
            result_keywords = set(self._extract_keywords(result_text))
            
            overlap = len(query_keywords & result_keywords)
            keyword_score = overlap / len(query_keywords) if query_keywords else 0.0
            keyword_scores.append(keyword_score)
        
        combined_scores = [(tfidf_scores[i] * 0.6 + keyword_scores[i] * 0.4) for i in range(len(search_results))]
        return combined_scores
    
    def _calculate_tfidf_similarity(self, query: str, documents: List[str]) -> List[float]:
        """Tính TF-IDF similarity."""
        try:
            corpus = [query] + documents
            
            vectorizer = TfidfVectorizer(
                stop_words=list(self.stop_words_vi),
                ngram_range=(1, 2),
                max_features=5000
            )
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            return similarities.tolist()
            
        except Exception as e:
            print(f"TF-IDF calculation error: {e}")
            return [0.5] * len(documents)
    
    def _calculate_quality_score(self, result: SearchResult) -> float:
        """Tính quality score dựa trên nhiều yếu tố."""
        score = 0.5
        
        if result.content:
            content_length = len(result.content)
            if content_length > 5000:
                score += 0.2
            elif content_length > 500:
                score += 0.1
        
        if result.title:
            if len(result.title) > 20 and len(result.title) < 100:
                score += 0.1
            if not any(spam_word in result.title.lower() for spam_word in ['click', 'free', 'buy', 'sale']):
                score += 0.1
        
        if result.url:
            if result.url.startswith('https://'):
                score += 0.05
            
            suspicious_patterns = ['bit.ly', 'tinyurl', 'short']
            if not any(pattern in result.url.lower() for pattern in suspicious_patterns):
                score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_freshness_score(self, result: SearchResult) -> float:
        """Calculate freshness score based on published date if available."""
        # Default score if no published date is available
        return 0.5
    
    def _calculate_authority_score(self, result: SearchResult) -> float:
        """Tính authority score dựa trên domain."""
        if not result.url:
            return 0.3
        
        trusted_domains = {
            'wikipedia.org': 0.95,
            'stackoverflow.com': 0.9,
            'github.com': 0.9,
            'medium.com': 0.85,
            'towardsdatascience.com': 0.85,
            'baidu.com':0.9,
            '.qq.com':0.97,
            '.edu': 0.9,
            '.wiki':0.9,
            '.gov': 0.9,
            '.org': 0.8,
            '.com': 0.7,
            '.vn': 0.7,
            '.com.vn': 0.7,
        }
        
        url_lower = result.url.lower()
        
        for domain, score in trusted_domains.items():
            if domain in url_lower:
                return score
        
        return 0.5
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized
    
    def _clean_text(self, text: str) -> str:
        """Clean và chuẩn hóa text."""
        if not text:
            return ""
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.lower().strip()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords từ text."""
        text_clean = self._clean_text(text)
        words = text_clean.split()
        
        keywords = [word for word in words 
                   if word not in self.stop_words_vi and len(word) > 2]
        
        return keywords 