"""
Agente Principal - Coordenador central do OTTO

Este módulo implementa o agente principal que coordena todos os componentes do OTTO,
processando consultas e direcionando para as ferramentas apropriadas.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class OTTOAgent:
    """Agente principal que coordena ferramentas e responde consultas."""
    
    def __init__(self, llm_manager, vector_store_manager, memory_manager):
        self.llm_manager = llm_manager
        self.vector_store_manager = vector_store_manager
        self.memory_manager = memory_manager
        
        # O vector store principal será carregado sob demanda na função de chat.
        # Não é necessário carregar aqui, pois o get_store já faz isso.
        
        logger.info("OTTOAgent inicializado.")
    
    def process_query(
        self,
        user_id: str,
        message: str,
        conversation_id: str = None,
        conversation_date: str = None
    ) -> Dict[str, Any]:
        """
        Processa uma consulta do usuário, orquestrando memória, RAG e LLM.
        
        Args:
            user_id (str): ID do usuário
            message (str): Mensagem do usuário
            conversation_id (str): ID da conversa (opcional)
            conversation_date (str): Data da conversa para busca (opcional)
        
        Returns:
            Dict: Resposta estruturada
        """
        try:
            logger.info(f"Processando mensagem do usuário {user_id}: {message[:50]}...")
            
            # 1. Resolver conversa (nova ou existente) - Reutilizando lógica do chat.py
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
            
            # 2. Obter contexto da conversa (Histórico)
            context = self._get_conversation_context(user_id, conversation_id)
            
            # 3. Processar a consulta (Chat Geral com RAG)
            response = self._handle_chat_geral(message, context, conversation_id)
            
            # 4. Salvar no histórico
            save_success = self.memory_manager.add_message_with_intent(
                conversation_id=conversation_id,
                user_message=message,
                assistant_response=response["content"],
                detected_intent="chat_geral", # Intent fixo após remoção do IntentDetector
                intent_confidence=1.0,
                metadata={
                    "user_id": user_id,
                    "chat_type": "general",
                    "knowledge_used": response.get("knowledge_used", False),
                    "sources": list(set(response.get("sources", [])))
                }
            )
            
            if not save_success:
                logger.warning(f"Falha ao salvar mensagem no histórico para conversa {conversation_id}")
            
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
                    "sources": list(set(response.get("sources", []))),
                    "response_type": response.get("type", "chat"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem no OTTOAgent: {e}")
            return {
                "success": False,
                "error": f"Erro interno: {str(e)}",
                "user_id": user_id,
                "conversation_id": conversation_id
            }

    def _resolve_conversation(
        self,
        user_id: str,
        conversation_id: str = None,
        conversation_date: str = None
    ) -> Dict[str, Any]:
        """
        Resolve qual conversa usar (nova ou existente) - Lógica copiada do chat.py
        """
        # Simplificação: Apenas cria uma nova conversa se não houver ID
        if conversation_id:
            metadata = self.memory_manager.get_conversation_metadata(conversation_id, user_id)
            if metadata:
                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "is_new": False
                }
            else:
                # Se o ID foi passado mas não existe, cria uma nova.
                conversation_id = None
        
        # Criar nova conversa
        new_conversation_id = self.memory_manager.create_conversation(
            user_id=user_id,
            conversation_title="Nova Conversa",
            conversation_type="chat"
        )
        
        return {
            "success": True,
            "conversation_id": new_conversation_id,
            "is_new": True
        }

    def _get_conversation_context(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Obtém contexto da conversa (Histórico)
        """
        try:
            # Obter histórico recente
            history = self.memory_manager.get_conversation_history(
                conversation_id=conversation_id,
                user_id=user_id,
                max_messages=10, # Usando valor fixo
                format_for_llm=True
            )
            
            return {
                "history": history,
                "message_count": len(history)
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter contexto: {e}")
            return {
                "history": [],
                "message_count": 0
            }

    def _handle_chat_geral(self, message: str, context: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
        """
        Função chat educacional geral com RAG
        """
        try:
            
            # 1. Buscar conhecimento relevante na base principal
            search_result = self.vector_store_manager.search(
                collection_name="main",
                query=message,
                n_results=5 # Usando valor fixo
            )

            
            knowledge_context = ""
            sources = []
            knowledge_used = False
            
            if search_result["success"] and search_result["results"]:
                # Filtrar resultados com boa relevância
                relevant_results = search_result["results"] # Em tese
                
                if relevant_results:
                    knowledge_context = "\n\n".join([
                        f"Informação relevante: {result['content']}"
                        for result in relevant_results
                    ])
                    sources = [result["metadata"].get("source", "Base de conhecimento") for result in relevant_results]
                    knowledge_used = True

            logger.info(f"CONTEXTO DE CONHECIMENTO: {knowledge_context}...")

            # 2. Gerar resposta educacional
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
                logger.error(f"Falha na resposta do LLM: {llm_response.get('error', 'Erro desconhecido')}")
                return {
                    "content": "Desculpe, ocorreu um erro ao processar sua mensagem. Tente novamente.",
                    "type": "error",
                    "knowledge_used": False,
                    "sources": [],
                    "error": llm_response.get('error', 'Erro desconhecido')
                }
                
        except Exception as e:
            logger.error(f"Erro no chat geral do OTTOAgent: {e}")
            return {
                "content": "Desculpe, ocorreu um erro interno. Tente novamente.",
                "type": "error",
                "knowledge_used": False,
                "sources": []
            }
