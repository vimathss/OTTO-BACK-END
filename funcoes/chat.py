"""
Chat Principal do OTTO

Esta função implementa o chat conversacional principal do OTTO.
NOTA: A lógica de orquestração foi movida para OTTOAgent.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Importar componentes do OTTO
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_manager import LLMManager
from core.vector_store import VectorStoreManager
from core.memory_manager import FirebaseMemoryManager
from core.intent_detector import IntentDetector # Mantido para compatibilidade de importação

logger = logging.getLogger(__name__)

class ChatPrincipal:
    """
    Classe principal para o chat conversacional do OTTO
    NOTA: Esta classe é mantida para compatibilidade, mas a lógica principal
    deve ser executada pelo OTTOAgent.
    """
    
    def __init__(self):
        """
        Inicializa o ChatPrincipal com todos os componentes necessários
        """
        try:
            # Inicializar componentes
            self.llm_manager = LLMManager()
            self.vector_store_manager = VectorStoreManager()
            self.memory_manager = FirebaseMemoryManager()
            self.intent_detector = IntentDetector() # Inicializado, mas desativado
            
            # Configurações
            self.config = {
                "max_context_messages": 10,
                "knowledge_search_limit": 5,
                "intent_confidence_threshold": 0.6,
                "default_chat_type": "general"
            }
            
            logger.info("ChatPrincipal inicializado (Lógica principal movida para OTTOAgent)")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar ChatPrincipal: {e}")
            raise
    
    def processar_mensagem(
        self,
        user_id: str,
        message: str,
        conversation_id: str = None,
        conversation_date: str = None,
        chat_type: str = None
    ) -> Dict[str, Any]:
        """
        Processa uma mensagem do usuário no chat principal
        
        NOTA: Esta função foi simplificada para ir direto ao chat geral,
        ignorando a detecção de intenção.
        """
        try:
            # 1. Resolver conversa (nova ou existente)
            resolved_conversation = self._resolve_conversation(
                user_id, conversation_id, conversation_date
            )
            
            if not resolved_conversation["success"]:
                return {
                    "success": False,
                    "error": resolved_conversation["error"],
                    "user_id": user_id
                }
            
            conversation_id = resolved_conversation["conversation_id"]
            is_new_conversation = resolved_conversation["is_new"]
            
            # 2. Obter contexto da conversa
            context = self._get_conversation_context(user_id, conversation_id)
            
            # 3. Processar como chat geral (com RAG)
            response = self._handle_chat_geral(message, context)
            
            # 4. Salvar no histórico
            save_success = self.memory_manager.add_message_with_intent(
                conversation_id=conversation_id,
                user_message=message,
                assistant_response=response["content"],
                detected_intent="chat_geral", # Fixo
                intent_confidence=1.0, # Fixo
                metadata={
                    "user_id": user_id,
                    "chat_type": chat_type or self.config["default_chat_type"],
                    "knowledge_used": response.get("knowledge_used", False),
                    "sources": response.get("sources", [])
                }
            )
            
            # 5. Preparar resposta final
            return {
                "success": True,
                "content": response["content"],
                "conversation_id": conversation_id,
                "user_id": user_id,
                "intent": {
                    "detected": "chat_geral",
                    "confidence": 1.0,
                    "matched_patterns": []
                },
                "metadata": {
                    "is_new_conversation": is_new_conversation,
                    "knowledge_used": response.get("knowledge_used", False),
                    "sources": response.get("sources", []),
                    "response_type": response.get("type", "chat"),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
            return {
                "success": False,
                "error": f"Erro interno: {str(e)}",
                "user_id": user_id,
                "conversation_id": conversation_id
            }
    
    # Funções auxiliares (mantidas para compatibilidade, mas a lógica principal está no OTTOAgent)
    def _resolve_conversation(self, user_id: str, conversation_id: str = None, conversation_date: str = None) -> Dict[str, Any]:
        # Lógica simplificada de resolução de conversa
        if conversation_id:
            metadata = self.memory_manager.get_conversation_metadata(conversation_id, user_id)
            if metadata:
                return {"success": True, "conversation_id": conversation_id, "is_new": False}
        
        new_conversation_id = self.memory_manager.create_conversation(user_id=user_id, conversation_title="Nova Conversa", conversation_type="chat")
        return {"success": True, "conversation_id": new_conversation_id, "is_new": True}

    def _get_conversation_context(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        history = self.memory_manager.get_conversation_history(conversation_id=conversation_id, user_id=user_id, max_messages=self.config["max_context_messages"], format_for_llm=True)
        return {"history": history, "message_count": len(history)}
    
    def _handle_chat_geral(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Função chat educacional geral (com RAG)
        """
        try:
            # Buscar conhecimento relevante na base principal
            search_result = self.vector_store_manager.search(query=message, collection_name="main", n_results=self.config["knowledge_search_limit"])
            
            knowledge_context = ""
            sources = []
            knowledge_used = False
            
            if search_result["success"] and search_result["results"]:
                relevant_results = [
                    result for result in search_result["results"]
                    if result.get("distance", 1.0) < 0.7  # Threshold de relevância
                ]
                
                if relevant_results:
                    knowledge_context = "\n\n".join([
                        f"Informação relevante: {result['content']}"
                        for result in relevant_results
                    ])
                    sources = [result["metadata"].get("source", "Base de conhecimento") for result in relevant_results]
                    knowledge_used = True
            

            # Gerar resposta educacional
            if knowledge_used:
                llm_response = self.llm_manager.generate_with_knowledge(
                    prompt=message,
                    knowledge_context=knowledge_context,
                    context=context["history"]
                )
            else:
                llm_response = self.llm_manager.generate_chat_response(
                    prompt=message,
                    context=context["history"],
                    chat_type="general"
                )
            
            if llm_response["success"]:
                return {
                    "content": llm_response["content"],
                    "type": "chat_geral",
                    "knowledge_used": knowledge_used,
                    "sources": sources
                }
            else:
                return {
                    "content": "Desculpe, ocorreu um erro ao processar sua mensagem. Tente novamente.",
                    "type": "error",
                    "knowledge_used": False,
                    "sources": []
                }
                
        except Exception as e:
            logger.error(f"Erro no chat geral: {e}")
            return {
                "content": "Desculpe, ocorreu um erro interno. Tente novamente.",
                "type": "error",
                "knowledge_used": False,
                "sources": []
            }

# Função principal para uso pela API (mantida para compatibilidade, mas o main.py usa o OTTOAgent)
def processar_chat_principal(
    user_id: str,
    message: str,
    conversation_id: str = None,
    conversation_date: str = None,
    chat_type: str = None
) -> Dict[str, Any]:
    try:
        chat = ChatPrincipal()
        return chat.processar_mensagem(
            user_id=user_id,
            message=message,
            conversation_id=conversation_id,
            conversation_date=conversation_date,
            chat_type=chat_type
        )
    except Exception as e:
        logger.error(f"Erro na função principal do chat: {e}")
        return {
            "success": False,
            "error": f"Erro interno: {str(e)}",
            "user_id": user_id
        }
