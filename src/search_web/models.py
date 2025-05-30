from dataclasses import dataclass

@dataclass
class SearchResult:
    title: str
    url: str
    description: str
    content: str = ""
    source: str = ""
    
@dataclass
class RankedResult:
    """Kết quả đã được rerank."""
    original_result: SearchResult
    relevance_score: float
    quality_score: float
    combined_score: float
    rank_position: int 