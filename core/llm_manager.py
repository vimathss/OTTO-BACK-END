"""
LLM Manager - Gerenciador da API do Google Gemini

Este módulo encapsula a lógica de comunicação com a API do Google Gemini,
incluindo a formatação de prompts e o tratamento de diferentes tipos de chat.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Gerenciador de comunicação com o modelo Gemini.
    """
    
    def __init__(self):
        """
        Inicializa o cliente Gemini.
        """
        try:
            # A chave da API deve ser definida como variável de ambiente (GEMINI_API_KEY)
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY não encontrada nas variáveis de ambiente")
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-2.5-flash" # Modelo gratuito e rápido
            logger.info(f"LLMManager inicializado com modelo: {self.model_name}")
        except Exception as e:
            logger.error(f"Erro ao inicializar o cliente Gemini: {e}")
            self.client = None
            
        # Configurações de sistema para diferentes tipos de chat
        self.system_prompts = {
            "general": "Você é o assistente educacional OTTO, um especialista em educação, BNCC, ABNT e elaboração de planos de aula. Responda de forma clara, educativa e profissional.",
        }

    def _format_history_for_gemini(self, history: List[Dict[str, str]]) -> List[types.Content]:
        """
        Converte o histórico de mensagens (role/content) para o formato Gemini (Content).
        O Gemini espera que o histórico seja alternado entre 'user' e 'model'.
        """
        formatted_history = []
        for message in history:
            role = message.get("role")
            content = message.get("content")
            
            if role == "user":
                formatted_history.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
            elif role == "assistant":
                # O Gemini usa 'model' para respostas do assistente
                formatted_history.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))
                
        return formatted_history

    def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Função central para gerar resposta do LLM.
        
        Args:
            prompt (str): A pergunta ou instrução principal do usuário.
            context (List[Dict]): Histórico da conversa (role/content).
            system_prompt (str): Instrução de sistema para o modelo.
            config (Dict): Configurações de geração (ex: temperature).
            
        Returns:
            Dict: Resposta estruturada (success, content, error).
        """
        if not self.client:
            return {"success": False, "error": "Cliente Gemini não inicializado."}
            
        try:
            # 1. Configurações de Geração
            generation_config = types.GenerateContentConfig(
                temperature=config.get("temperature", 0.7) if config else 0.7
            )
            
            # 2. Conteúdo da Requisição
            contents = []
            
            # Adicionar histórico (contexto)
            if context:
                contents.extend(self._format_history_for_gemini(context))
            
            # Adicionar a mensagem atual do usuário
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
            
            # 3. Instrução de Sistema
            # Se um system_prompt for fornecido, ele tem prioridade
            if system_prompt:
                system_instruction = system_prompt
            else:
                # Se não, usa o prompt geral
                system_instruction = self.system_prompts["general"]
            
            # 4. Chamada à API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generation_config,

            )
            
            return {"success": True, "content": response.text}
            
        except APIError as e:
            logger.error(f"Erro da API Gemini: {e}")
            return {"success": False, "error": f"Erro da API Gemini: {e}"}
        except Exception as e:
            logger.error(f"Erro desconhecido na geração: {e}")
            return {"success": False, "error": f"Erro desconhecido: {e}"}

    def generate_chat_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        chat_type: str = "general",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gera resposta para um chat sem RAG.
        """
        system_prompt = self.system_prompts.get(chat_type, self.system_prompts["general"])
        
        return self.generate_response(
            prompt=prompt,
            context=context,
            system_prompt=system_prompt,
            config=config
        )

    def generate_with_knowledge(
        self,
        prompt: str,
        knowledge_context: str,
        context: Optional[List[Dict[str, str]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gera resposta usando RAG (Retrieval-Augmented Generation).
        """

        # O prompt é enriquecido com o contexto de conhecimento
        rag_prompt = f"""
        INFORMAÇÕES DE CONTEXTO (RAG):
        {knowledge_context}
        
        PERGUNTA DO USUÁRIO:
        {prompt}
        
        Instrução: Use as INFORMAÇÕES DE CONTEXTO para responder à PERGUNTA DO USUÁRIO. Se as informações de contexto não forem suficientes ou relevantes, use seu conhecimento geral, mas priorize o contexto fornecido. Mantenha a resposta clara, educativa e profissional.
        """
        
        # Usar o system prompt geral para manter o persona do OTTO
        system_prompt = self.system_prompts["general"]

        logger.info(f"RAG PROMPT GERADO: {rag_prompt}")

        
        return self.generate_response(
            prompt=rag_prompt,
            context=context,
            system_prompt=system_prompt,
            config=config
        )

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Função de compatibilidade para chamadas simples (ex: redacao_tool).
        """
        response = self.generate_response(
            prompt=prompt,
            context=None,
            system_prompt=None,
            config={"temperature": temperature}
        )
        
        if response["success"]:
            return response["content"]
        else:
            logger.error(f"Falha na geração simples: {response['error']}")
            return f"Erro na geração: {response['error']}"