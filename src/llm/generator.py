import ollama


class OllamaServiceError(RuntimeError):
    """Raised when the local Ollama service cannot complete an operation."""


class OllamaModelUnavailableError(OllamaServiceError):
    """Raised when the configured model is not installed in Ollama."""


class LLMGenerator:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.model_name = model_name

    def ensure_available(self) -> None:
        """Check that Ollama is reachable and has the configured model."""
        try:
            response = ollama.list()
            models = (
                response.get("models", [])
                if isinstance(response, dict)
                else response.models
            )
            model_names = []
            for model in models:
                if isinstance(model, dict):
                    model_name = model.get("name") or model.get("model")
                else:
                    model_name = getattr(model, "name", None) or getattr(
                        model, "model", None
                    )
                if isinstance(model_name, str):
                    model_names.append(model_name)

            if self.model_name not in model_names:
                raise OllamaModelUnavailableError(
                    f"Ollama model '{self.model_name}' is not available."
                )
        except OllamaModelUnavailableError:
            raise
        except Exception as error:
            raise OllamaServiceError("Ollama preflight failed.") from error

    def generate(self, prompt: str) -> str:
        """Generate once and tell Ollama to unload the model afterwards."""
        self.ensure_available()
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là một trợ lý pháp lý AI thông minh tại Việt Nam. Hãy trả lời dựa trên các thông tin pháp luật được cung cấp, ngắn gọn và chính xác."},
                    {"role": "user", "content": prompt},
                ],
                keep_alive=0,
            )
            return response["message"]["content"]
        except Exception as error:
            raise OllamaServiceError("Ollama generation failed.") from error

# ---- ĐOẠN CODE TEST THỬ ----
if __name__ == "__main__":
    llm = LLMGenerator()
    cau_hoi = "Hợp đồng lao động miệng có giá trị pháp lý không?"
    print(f"\nHỏi: {cau_hoi}")
    print("AI đang suy nghĩ...")
    tra_loi = llm.generate(cau_hoi)
    print(f"\nĐáp án:\n{tra_loi}")
