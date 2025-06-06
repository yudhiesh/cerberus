import os
from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI
import json
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter, stop_after_attempt

retryable_exceptions = (
    Exception,  # You might need to update these based on the new client exceptions
)

class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        *args,
        **kwargs,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base = api_base or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        self.temperature = temperature
        self.args = args
        self.kwargs = kwargs
        
        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided via argument or OPENROUTER_API_KEY env var.")
        
        # Create the OpenAI client with custom base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
        super().__init__(model_name)
    
    def get_model_name(self):
        return f"OpenRouter ({self.model_name})"
    
    def load_model(self, async_mode: bool = False):
        # Return the configured client
        return self.client
    
    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)
        
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        
        if schema:
            # Try to use function calling/JSON mode if supported
            kwargs["response_format"] = {"type": "json_object"}
        
        # Use the client instance to make the API call
        response = client.chat.completions.create(**kwargs)
        
        output = response.choices[0].message.content
        
        if schema:
            try:
                json_output = json.loads(output)
                return schema.model_validate(json_output), 0.0
            except Exception:
                return output, 0.0
        
        return output, 0.0
    
    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        # For now, just call sync generate (can be improved with openai AsyncClient)
        return self.generate(prompt, schema)