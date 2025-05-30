import requests

class GeminiAPI:
    """Class để kết nối với Gemini API thực tế."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def generate_answer(self, prompt: str) -> str:
        """Tạo câu trả lời từ Gemini API."""
        try:
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "Không thể tạo câu trả lời từ Gemini API."
            else:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return f"Lỗi API: {response.status_code}. Vui lòng kiểm tra API key hoặc thử lại sau."
                
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"Lỗi khi gọi Gemini API: {str(e)}" 