import os
from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

class OpenRouterOpenAI(DeepEvalBaseLLM):
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize by reading OPENAI_API_KEY and OPENAI_API_BASE from environment.
        
        - model_name: e.g. "gpt-4o" or "gpt-3.5-turbo"
        - kwargs: any additional OpenAI parameters (temperature, max_tokens, etc.)
        """

        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        if api_key is None or api_base is None:
            raise ValueError(
                "Both OPENAI_API_KEY and OPENAI_API_BASE must be set in the environment"
            )

        self.client = OpenAI(
            api_key   = api_key,
            base_url  = api_base.rstrip("/")
        )
        self.model_name     = model_name
        self.request_kwargs = kwargs  # e.g., {"temperature": 0.0, "max_tokens": 256}

    def load_model(self):
        """
        Return a dict of parameters for chat completion.
        Since the OpenAI SDK v1.0.0+ has no persistent ChatCompletion object,
        we return the dict that will be unpacked into the create/acreate call.
        """
        return {
            "model": self.model_name,
            **self.request_kwargs
        }  # :contentReference[oaicite:11]{index=11}

    def generate(self, prompt: str) -> str:
        """
        Synchronous generation using `client.chat.completions.create(...)`.
        Returns the assistant’s content.
        """
        params = self.load_model()
        # 3) Call the new create(...) method under chat.completions
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **params
        )  # :contentReference[oaicite:12]{index=12}

        # 4) Extract and return the assistant’s reply text
        return response.choices[0].message.content.strip()  # :contentReference[oaicite:13]{index=13}

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronous generation using `client.chat.completions.acreate(...)`.
        Returns the assistant’s content.
        """
        params = self.load_model()
        response = await self.client.chat.completions.acreate(
            messages=[{"role": "user", "content": prompt}],
            **params
        )  

        return response.choices[0].message.content.strip()  # :

    def get_model_name(self):
        return f"OpenRouter/OpenAI ({self.model_name})"
