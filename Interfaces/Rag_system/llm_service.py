import logging
import requests
import json
import re
from typing import Dict, Any, Optional, List, Union
import config

# if config.LLM_LOCAL_MODE:
#     from unsloth import FastLanguageModel
#     from unsloth.chat_templates import get_chat_template

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM inference."""

    def __init__(self,
                 service_url=config.LLM_SERVICE_URL):
        """Initialize the LLM service."""

        self.service_url = service_url

    # def __init__(self, local_mode=config.LLM_LOCAL_MODE, model_name=config.LLM_MODEL_NAME,
    #              service_url=config.LLM_SERVICE_URL):
    #     """Initialize the LLM service."""
    #     self.local_mode = local_mode
    #     self.service_url = service_url

        # if local_mode:
        #     logger.info(f"Initializing local LLM with model {model_name}")
        #     try:
        #         self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        #             model_name=model_name,
        #             max_seq_length=6096,
        #             load_in_4bit=True
        #         )
        #
        #         FastLanguageModel.for_inference(self.model)
        #
        #         # Apply chat template
        #         if "phi" in model_name.lower():
        #             template = "phi-3"
        #         elif "qwen" in model_name.lower():
        #             template = "qwen-2.5"
        #         elif "mistral" in model_name.lower():
        #             template = "mistral"
        #         else:
        #             template = "llama-3"
        #
        #         self.tokenizer = get_chat_template(
        #             self.tokenizer,
        #             chat_template=template,
        #         )
        #
        #         logger.info(f"Local LLM initialized with {template} template")
        #     except Exception as e:
        #         logger.error(f"Failed to initialize local LLM: {e}")
        #         raise
        # else:
        #     logger.info(f"Using remote LLM service at {service_url}")

    # def generate(self, prompt: Union[str, List[Dict[str, str]]],
    #              system_prompt: Optional[str] = None,
    #              temperature: float = 0.4,
    #              max_tokens: int = 1024,
    #              top_p: float = 0.9,
    #              repetition_penalty: float = 1.1) -> str:
    #     """
    #     Generate text from the LLM.
    #     """
    #     if self.local_mode:
    #         return self._generate_local(prompt, system_prompt, temperature, max_tokens, top_p, repetition_penalty)
    #     else:
    #         return self._generate_remote(prompt, system_prompt, temperature, max_tokens, top_p, repetition_penalty)

    def generate(self, prompt: Union[str, List[Dict[str, str]]],
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.4,
                 max_tokens: int = 1024,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1) -> str:
        """
        Generate text from the LLM.
        """
        return self._generate_remote(prompt, system_prompt, temperature, max_tokens, top_p, repetition_penalty)

    # def _generate_local(self, prompt: Union[str, List[Dict[str, str]]],
    #                     system_prompt: Optional[str] = None,
    #                     temperature: float = 0.4,
    #                     max_tokens: int = 1024,
    #                     top_p: float = 0.9,
    #                     repetition_penalty: float = 1.1) -> str:
    #     """Generate text using the local LLM."""
    #     try:
    #         # Prepare messages
    #         if isinstance(prompt, str):
    #             messages = []
    #             if system_prompt:
    #                 messages.append({"role": "system", "content": system_prompt})
    #             messages.append({"role": "user", "content": prompt})
    #         else:
    #             # Prompt is already a list of messages
    #             messages = prompt
    #             if system_prompt and not (messages and messages[0]["role"] == "system"):
    #                 messages = [{"role": "system", "content": system_prompt}] + messages
    #
    #         # Tokenize input
    #         inputs = self.tokenizer.apply_chat_template(
    #             messages,
    #             tokenize=True,
    #             add_generation_prompt=True,
    #             return_tensors="pt"
    #         ).to("cuda")
    #
    #         # Generate output
    #         output_ids = self.model.generate(
    #             input_ids=inputs,
    #             max_new_tokens=max_tokens,
    #             temperature=temperature,
    #             top_p=top_p,
    #             repetition_penalty=repetition_penalty
    #         )
    #
    #         # Decode output
    #         prompt_length = inputs.shape[1]
    #         generated_text = self.tokenizer.decode(
    #             output_ids[0][prompt_length:],
    #             skip_special_tokens=True
    #         ).strip()
    #
    #         return generated_text
    #
    #     except Exception as e:
    #         logger.error(f"Error during local generation: {e}")
    #         return f"Error generating response: {str(e)}"

    def _generate_remote(self, prompt: Union[str, List[Dict[str, str]]],
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.4,
                         max_tokens: int = 3024,
                         top_p: float = 0.9,
                         repetition_penalty: float = 1.1,
                         remove_thinking: bool = True) -> str:
        """Generate text using the remote LLM service."""
        try:
            # Prepare request payload
            payload = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "remove_thinking": remove_thinking
            }

            if isinstance(prompt, str):
                payload["prompt"] = prompt
                if system_prompt:
                    payload["system_prompt"] = system_prompt
            else:
                payload["messages"] = prompt

            # Make API request
            response = requests.post(
                f"{self.service_url}/generate",
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"Remote LLM service error: {response.status_code} - {response.text}")
                return f"Error from LLM service: {response.text}"

            # Parse response
            result = response.json()
            return result.get("text", "No text generated")

        except requests.RequestException as e:
            logger.error(f"Error connecting to remote LLM service: {e}")
            return f"Error connecting to LLM service: {str(e)}"
        except Exception as e:
            logger.error(f"Error during remote generation: {e}")
            return f"Error generating response: {str(e)}"

    def extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from generated text.
        """
        try:
            # Try to find JSON pattern in the text
            json_match = re.search(r'({.*})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)

            # If no JSON found in {}
            json_match = re.search(r'(\[.*\])', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)

            return json.loads(text)

        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return {}


llm_service = LLMService()
