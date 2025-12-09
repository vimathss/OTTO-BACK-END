"""
MemoryManager com Firebase Cloud Firestore

Este módulo gerencia o histórico de conversas do OTTO utilizando o Firebase Cloud Firestore
como backend de persistência, permitindo armazenamento remoto e escalável.
"""

import os
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

logger = logging.getLogger(__name__)

class FirebaseMemoryManager:
    """
    Gerenciador de memória utilizando Firebase Cloud Firestore
    """
    
    def __init__(self, credentials_path: str = None):
        """
        Inicializa o MemoryManager com Firebase
        
        Args:
            credentials_path (str): Caminho para o arquivo de credenciais do Firebase
        """
        self.db = None
        self._initialize_firebase(credentials_path)
        
        # Configurações
        self.max_messages_per_conversation = 1000
        self.max_conversations_per_user = 100
        
        logger.info("FirebaseMemoryManager inicializado com sucesso")
    
    def _initialize_firebase(self, credentials_path: str = None):
        """
        Inicializa a conexão com o Firebase
        """
        try:
            # Usar credenciais do arquivo ou variável de ambiente
            if not credentials_path:
                credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
            
            # Verificar se o arquivo de credenciais existe
            if not os.path.exists(credentials_path):
                # ATENÇÃO: Se o arquivo não existir, o sistema não funcionará.
                # Para o sandbox, vamos apenas logar um aviso.
                logger.warning(f"Arquivo de credenciais não encontrado: {credentials_path}. A memória não funcionará.")
                return
            
            # Inicializar Firebase Admin SDK (apenas se ainda não foi inicializado)
            if not firebase_admin._apps:
                cred = credentials.Certificate(credentials_path)
                firebase_admin.initialize_app(cred)
                logger.info(f"Firebase inicializado com credenciais: {credentials_path}")
            else:
                logger.info("Firebase já estava inicializado")
            
            # Obter cliente do Firestore
            self.db = firestore.client()
            logger.info("Cliente Firestore conectado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar Firebase: {e}")
            # Não levantar exceção, apenas logar, para não quebrar o __init__ se for um problema de credenciais
            # O erro será capturado nas funções que tentam usar self.db
            pass
    
    def create_conversation(
        self,
        user_id: str,
        conversation_title: str = None,
        conversation_type: str = "chat"
    ) -> str:
        """
        Cria uma nova conversa
        
        Args:
            user_id (str): ID do usuário
            conversation_title (str): Título da conversa
            conversation_type (str): Tipo da conversa
        
        Returns:
            str: ID da conversa criada
        """
        if not self.db:
            raise ConnectionError("Firebase não inicializado. Verifique as credenciais.")
            
        try:
            # Gerar ID da conversa baseado na data e hora atual
            now = datetime.now(timezone.utc)
            conversation_id = now.strftime("%Y-%m-%d_%H-%M-%S")
            
            # Se não foi fornecido um título, gerar um baseado na data
            if not conversation_title:
                conversation_title = f"Conversa {now.strftime('%d/%m/%Y às %H:%M')}"
            
            # Dados da conversa
            conversation_data = {
                "conversation_id": conversation_id,
                "title": conversation_title,
                "type": conversation_type,
                "created_at": now,
                "updated_at": now,
                "user_id": user_id,
                "messages": [],
                "message_count": 0
            }
            
            # Salvar no Firestore
            doc_ref = self.db.collection("users").document(user_id).collection("conversations").document(conversation_id)
            doc_ref.set(conversation_data)
            
            logger.info(f"Conversa criada: {conversation_id} para usuário {user_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Erro ao criar conversa: {e}")
            raise
    
    def add_message_with_intent(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        detected_intent: str = None,
        intent_confidence: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Adiciona uma mensagem à conversa com informações de intenção
        
        Args:
            conversation_id (str): ID da conversa
            user_message (str): Mensagem do usuário
            assistant_response (str): Resposta do assistente
            detected_intent (str): Intenção detectada
            intent_confidence (float): Confiança da detecção de intenção
            metadata (Dict): Metadados adicionais
        
        Returns:
            bool: True se a mensagem foi adicionada com sucesso
        """
        if not self.db:
            logger.error("Firebase não inicializado. Não foi possível adicionar mensagem.")
            return False
            
        try:
            # Extrair user_id do metadata ou usar padrão
            user_id = metadata.get("user_id", "unknown") if metadata else "unknown"
            
            # Buscar a conversa
            doc_ref = self.db.collection("users").document(user_id).collection("conversations").document(conversation_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                logger.warning(f"Conversa {conversation_id} não encontrada para usuário {user_id}")
                return False
            
            conversation_data = doc.to_dict()
            
            # Preparar metadados da mensagem
            message_metadata = metadata.copy() if metadata else {}
            if detected_intent:
                message_metadata["detected_intent"] = detected_intent
            if intent_confidence is not None:
                message_metadata["intent_confidence"] = intent_confidence
            
            # Criar timestamp
            now = datetime.now(timezone.utc)
            
            # Criar mensagens do usuário e do assistente
            user_msg = {
                "timestamp": now,
                "role": "user",
                "content": user_message,
                "metadata": {}
            }
            
            assistant_msg = {
                "timestamp": now,
                "role": "assistant", 
                "content": assistant_response,
                "metadata": message_metadata
            }
            
            # Adicionar mensagens ao array existente
            messages = conversation_data.get("messages", [])
            messages.extend([user_msg, assistant_msg])
            
            # Limitar número de mensagens se necessário
            if len(messages) > self.max_messages_per_conversation:
                messages = messages[-self.max_messages_per_conversation:]
            
            # Atualizar documento
            doc_ref.update({
                "messages": messages,
                "message_count": len(messages),
                "updated_at": now
            })
            
            logger.info(f"Mensagem adicionada à conversa {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao adicionar mensagem: {e}")
            return False
    
    def get_conversation_history(
        self,
        conversation_id: str,
        user_id: str = None,
        max_messages: int = 50,
        format_for_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Obtém o histórico de uma conversa
        
        Args:
            conversation_id (str): ID da conversa
            user_id (str): ID do usuário (se conhecido)
            max_messages (int): Número máximo de mensagens
            format_for_llm (bool): Se deve formatar para uso com LLM
        
        Returns:
            List[Dict]: Lista de mensagens da conversa
        """
        if not self.db:
            logger.error("Firebase não inicializado. Não foi possível obter histórico.")
            return []
            
        try:
            # Se user_id não foi fornecido, tentar encontrar a conversa em todos os usuários
            if not user_id:
                user_id = self._find_user_by_conversation(conversation_id)
                if not user_id:
                    logger.warning(f"Conversa {conversation_id} não encontrada")
                    return []
            
            # Buscar a conversa
            doc_ref = self.db.collection("users").document(user_id).collection("conversations").document(conversation_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                logger.warning(f"Conversa {conversation_id} não encontrada para usuário {user_id}")
                return []
            
            conversation_data = doc.to_dict()
            messages = conversation_data.get("messages", [])
            
            # Limitar número de mensagens
            if max_messages and len(messages) > max_messages:
                messages = messages[-max_messages:]
            
            # Formatar para LLM se solicitado
            if format_for_llm:
                formatted_messages = []
                for msg in messages:
                    # Converter timestamp para string se for um objeto datetime
                    if isinstance(msg.get("timestamp"), datetime):
                        msg["timestamp"] = msg["timestamp"].isoformat()
                    
                    # CORREÇÃO: O histórico deve ser formatado como uma lista de objetos
                    # com 'role' e 'content' para ser usado pelo LLMManager
                    formatted_msg = {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                    formatted_messages.append(formatted_msg)
                return formatted_messages
            else:
                # Converter timestamps para string para retorno consistente
                for msg in messages:
                    if isinstance(msg.get("timestamp"), datetime):
                        msg["timestamp"] = msg["timestamp"].isoformat()
                return messages
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico da conversa: {e}")
            return []
    
    def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20,
        conversation_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém todas as conversas de um usuário
        
        Args:
            user_id (str): ID do usuário
            limit (int): Número máximo de conversas
            conversation_type (str): Filtrar por tipo de conversa
        
        Returns:
            List[Dict]: Lista de conversas do usuário
        """
        if not self.db:
            logger.error("Firebase não inicializado. Não foi possível obter conversas.")
            return []
            
        try:
            # Referência para a coleção de conversas do usuário
            conversations_ref = self.db.collection("users").document(user_id).collection("conversations")
            
            # Aplicar filtro por tipo se especificado
            if conversation_type:
                query = conversations_ref.where(filter=FieldFilter("type", "==", conversation_type))
            else:
                query = conversations_ref
            
            # Ordenar por data de atualização (mais recentes primeiro) e limitar
            query = query.order_by("updated_at", direction=firestore.Query.DESCENDING).limit(limit)
            
            # Executar query
            docs = query.stream()
            
            conversations = []
            for doc in docs:
                conv_data = doc.to_dict()
                # Incluir apenas metadados, não as mensagens completas
                conversation_summary = {
                    "conversation_id": conv_data.get("conversation_id"),
                    "title": conv_data.get("title"),
                    "type": conv_data.get("type"),
                    "created_at": conv_data.get("created_at").isoformat() if isinstance(conv_data.get("created_at"), datetime) else conv_data.get("created_at"),
                    "updated_at": conv_data.get("updated_at").isoformat() if isinstance(conv_data.get("updated_at"), datetime) else conv_data.get("updated_at"),
                    "message_count": conv_data.get("message_count", 0)
                }
                conversations.append(conversation_summary)
            
            logger.info(f"Encontradas {len(conversations)} conversas para usuário {user_id}")
            return conversations
            
        except Exception as e:
            logger.error(f"Erro ao obter conversas do usuário: {e}")
            return []
    
    def find_conversation_by_date(
        self,
        user_id: str,
        target_date: str
    ) -> Optional[str]:
        """
        Encontra uma conversa por data
        
        Args:
            user_id (str): ID do usuário
            target_date (str): Data no formato "YYYY-MM-DD" ou "DD/MM/YYYY"
        
        Returns:
            Optional[str]: ID da conversa encontrada ou None
        """
        if not self.db:
            logger.error("Firebase não inicializado. Não foi possível buscar conversa por data.")
            return None
            
        try:
            # Normalizar formato da data
            if "/" in target_date:
                # Converter DD/MM/YYYY para YYYY-MM-DD
                day, month, year = target_date.split("/")
                normalized_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                normalized_date = target_date
            
            # Buscar conversas do usuário
            conversations_ref = self.db.collection("users").document(user_id).collection("conversations")
            
            # Filtrar conversas que começam com a data especificada
            # Note: Firestore não suporta diretamente LIKE ou regex em queries. 
            # Usamos range queries para simular a busca por prefixo de data.
            query = conversations_ref.where(
                filter=FieldFilter("conversation_id", ">=", normalized_date)
            ).where(
                filter=FieldFilter("conversation_id", "<", normalized_date + "\uf8ff") # \uf8ff é um caractere unicode que garante o fim do range
            ).order_by("conversation_id").limit(1)
            
            docs = list(query.stream())
            
            if docs:
                # Retornar a primeira conversa encontrada (mais antiga do dia)
                first_conversation = docs[0].to_dict()
                conversation_id = first_conversation.get("conversation_id")
                logger.info(f"Conversa encontrada para data {target_date}: {conversation_id}")
                return conversation_id
            else:
                logger.info(f"Nenhuma conversa encontrada para data {target_date}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao buscar conversa por data: {e}")
            return None
    
    def delete_conversation(self, conversation_id: str, user_id: str = None) -> bool:
        """
        Deleta uma conversa
        
        Args:
            conversation_id (str): ID da conversa
            user_id (str): ID do usuário
        
        Returns:
            bool: True se a conversa foi deletada com sucesso
        """
        if not self.db:
            logger.error("Firebase não inicializado. Não foi possível deletar conversa.")
            return False
            
        try:
            # Se user_id não foi fornecido, tentar encontrar
            if not user_id:
                user_id = self._find_user_by_conversation(conversation_id)
                if not user_id:
                    logger.warning(f"Conversa {conversation_id} não encontrada para deletar")
                    return False
            
            # Deletar documento
            doc_ref = self.db.collection("users").document(user_id).collection("conversations").document(conversation_id)
            doc_ref.delete()
            
            logger.info(f"Conversa {conversation_id} deletada com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao deletar conversa: {e}")
            return False
    
    def get_recent_intents(
        self,
        conversation_id: str,
        user_id: str = None,
        limit: int = 5
    ) -> List[str]:
        """
        Obtém as intenções mais recentes de uma conversa
        
        Args:
            conversation_id (str): ID da conversa
            user_id (str): ID do usuário
            limit (int): Número máximo de intenções
        
        Returns:
            List[str]: Lista de intenções recentes
        """
        if not self.db:
            logger.error("Firebase não inicializado. Não foi possível obter intenções recentes.")
            return []
            
        try:
            # Obter histórico da conversa
            messages = self.get_conversation_history(
                conversation_id=conversation_id,
                user_id=user_id,
                max_messages=limit * 2,  # Buscar mais mensagens para garantir que temos intenções suficientes
                format_for_llm=False
            )
            
            # Extrair intenções das mensagens do assistente
            intents = []
            for msg in reversed(messages):  # Mais recentes primeiro
                if msg.get("role") == "assistant":
                    metadata = msg.get("metadata", {})
                    intent = metadata.get("detected_intent")
                    if intent and intent not in intents:
                        intents.append(intent)
                        if len(intents) >= limit:
                            break
            
            return intents
            
        except Exception as e:
            logger.error(f"Erro ao obter intenções recentes: {e}")
            return []
    
    def _find_user_by_conversation(self, conversation_id: str) -> Optional[str]:
        """
        Encontra o usuário proprietário de uma conversa
        
        Args:
            conversation_id (str): ID da conversa
            
        Returns:
            Optional[str]: ID do usuário ou None se não encontrado
        """
        if not self.db:
            return None
            
        try:
            # Esta é uma operação custosa - idealmente, sempre passe o user_id
            # Para otimização, você pode manter um índice separado de conversation_id -> user_id
            
            # Por enquanto, vamos buscar em todos os usuários (não recomendado para produção)
            users_ref = self.db.collection("users")
            users = users_ref.stream()
            
            for user_doc in users:
                user_id = user_doc.id
                conv_ref = self.db.collection("users").document(user_id).collection("conversations").document(conversation_id)
                if conv_ref.get().exists:
                    return user_id
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao encontrar usuário por conversa: {e}")
            return None
    
    def get_conversation_metadata(
        self,
        conversation_id: str,
        user_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Obtém apenas os metadados de uma conversa (sem as mensagens)
        
        Args:
            conversation_id (str): ID da conversa
            user_id (str): ID do usuário
        
        Returns:
            Optional[Dict]: Metadados da conversa ou None se não encontrada
        """
        if not self.db:
            logger.error("Firebase não inicializado. Não foi possível obter metadados.")
            return None
            
        try:
            if not user_id:
                user_id = self._find_user_by_conversation(conversation_id)
                if not user_id:
                    return None
            
            doc_ref = self.db.collection("users").document(user_id).collection("conversations").document(conversation_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return None
            
            conv_data = doc.to_dict()
            
            # Retornar apenas metadados, excluindo as mensagens
            metadata = {
                "conversation_id": conv_data.get("conversation_id"),
                "title": conv_data.get("title"),
                "type": conv_data.get("type"),
                "created_at": conv_data.get("created_at").isoformat() if isinstance(conv_data.get("created_at"), datetime) else conv_data.get("created_at"),
                "updated_at": conv_data.get("updated_at").isoformat() if isinstance(conv_data.get("updated_at"), datetime) else conv_data.get("updated_at"),
                "user_id": conv_data.get("user_id"),
                "message_count": conv_data.get("message_count", 0)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Erro ao obter metadados da conversa: {e}")
            return None
    
    def update_conversation_title(
        self,
        conversation_id: str,
        new_title: str,
        user_id: str = None
    ) -> bool:
        """
        Atualiza o título de uma conversa
        
        Args:
            conversation_id (str): ID da conversa
            new_title (str): Novo título
            user_id (str): ID do usuário
        
        Returns:
            bool: True se o título foi atualizado com sucesso
        """
        if not self.db:
            logger.error("Firebase não inicializado. Não foi possível atualizar título.")
            return False
            
        try:
            if not user_id:
                user_id = self._find_user_by_conversation(conversation_id)
                if not user_id:
                    return False
            
            doc_ref = self.db.collection("users").document(user_id).collection("conversations").document(conversation_id)
            doc_ref.update({
                "title": new_title,
                "updated_at": datetime.now(timezone.utc)
            })
            
            logger.info(f"Título da conversa {conversation_id} atualizado para: {new_title}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao atualizar título da conversa: {e}")
            return False

# Manter compatibilidade com o nome original
MemoryManager = FirebaseMemoryManager
