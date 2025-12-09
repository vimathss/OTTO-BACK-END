"""
RedacaoTool - Ferramenta de correção de redação para o OTTO

Este módulo implementa a lógica de correção, construindo prompts detalhados
e forçando a saída em formato JSON para facilitar o uso no Frontend.
"""

import os
import logging
import json
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RedacaoTool:
    """
    Ferramenta de correção de redação que utiliza RAG e LLM.
    """
    
    def __init__(self, llm_manager, vector_store_manager):
        self.llm_manager = llm_manager
        self.vector_store_manager = vector_store_manager
        self.redacao_store_name = "redacao_knowledge"
        
        # Tenta carregar o vector store (opcional para este passo)
        try:
            self.redacao_store = self.vector_store_manager.get_store(self.redacao_store_name)
            if not self.redacao_store:
                # Log de aviso se não houver base vetorial, mas continua funcionando só com LLM
                logger.warning(f"Vector store '{self.redacao_store_name}' não encontrado. A correção usará apenas o conhecimento do LLM.")
        except Exception as e:
            logger.error(f"Erro ao inicializar RedacaoTool: {e}")
            self.redacao_store = None
            
        logger.info("RedacaoTool inicializado.")

    def corrigir_redacao(
        self,
        user_id: str,
        essay_text: str,
        essay_type: str,       # ex: 'enem', 'fuvest'
        essay_title: str,
        student_name: str,
        student_class: str,
        teacher_comments: str
    ) -> Dict[str, Any]:
        """
        Corrige a redação e retorna um dicionário estruturado (JSON).
        """
        try:
            logger.info(f"Iniciando correção para {student_name} ({essay_type})")

            # 1. Busca RAG (Opcional - busca critérios específicos do tipo de prova)
            query_rag = f"Critérios de correção para redação tipo {essay_type}"
            knowledge_context = self._get_knowledge_context(query_rag)
            
            # 2. Constrói o Prompt Detalhado
            prompt = self._build_correction_prompt(
                essay_text, essay_type, essay_title, 
                student_name, student_class, teacher_comments, 
                knowledge_context
            )
            
            # 3. Chama o LLM
            # Temperature baixa (0.2) para ser mais analítico e menos "criativo"
            system_prompt = f"Você é um especialista rigoroso em correção de redações modelo {essay_type}. Analise a redação e forneça uma análise detalhada e construtiva."
            
            llm_response = self.llm_manager.generate_response(
                prompt=prompt,
                context=None, # Sem histórico de chat, é uma tarefa única
                system_prompt=system_prompt,
                config={"temperature": 0.2} 
            )
            
            if not llm_response["success"]:
                return {"success": False, "error": llm_response.get("error")}
            
            # 4. Processa a resposta (JSON Parsing)
            content_raw = llm_response["content"]
            parsed_content = self._parse_json_response(content_raw)
            
            # Se falhar no parse, retornamos erro
            if not parsed_content:
                return {
                    "success": False, 
                    "error": "Falha ao estruturar a resposta da IA. O modelo não retornou um JSON válido."
                }

            # 5. Retorna no formato que o Frontend espera (RedacaoResponse)
            # Expandimos (**parsed_content) para que chaves como 'nota', 'c1', etc. fiquem na raiz
            return {
                "success": True,
                "content": "Correção realizada com sucesso.", # Mensagem de sistema
                "user_id": user_id,
                "knowledge_used": bool(knowledge_context),
                "sources": [f"Critérios de Correção {essay_type.upper()}" if knowledge_context else "Conhecimento Geral do LLM"],
                **parsed_content 
            }

        except Exception as e:
            logger.error(f"Erro crítico na RedacaoTool: {e}")
            return {"success": False, "error": str(e)}

    def _get_knowledge_context(self, query: str) -> str:
        """Tenta buscar critérios de correção no Vector Store."""
        if not self.redacao_store:
            return ""
        try:
            # Busca simples por similaridade
            results = self.redacao_store.similarity_search(query, k=2)
            return "\n".join([doc.page_content for doc in results])
        except Exception:
            return ""

    def _build_correction_prompt(self, text, tipo, titulo, nome, sala, comentarios, knowledge) -> str:
        """
        Monta o prompt exigindo saída JSON estrita.
        """
        return f"""
        DADOS DO ALUNO:
        Nome: {nome} | Sala: {sala}
        Título da Redação: {titulo}
        
        PEDIDOS ESPECIAIS DO PROFESSOR:
        "{comentarios}"
        
        CRITÉRIOS DE CORREÇÃO (Referência):
        {knowledge}
        
        TEXTO DA REDAÇÃO:
        \"\"\"
        {text}
        \"\"\"
        
        ---
        
        INSTRUÇÕES DE SAÍDA (OBRIGATÓRIO):
        Analise a redação acima como um corretor oficial do {tipo.upper()}.
        Retorne APENAS um objeto JSON válido. Não escreva nada antes ou depois do JSON (sem ```json).
        
        A estrutura do JSON deve ser EXATAMENTE esta:
        {{
            "nota": "Uma string com a nota final (ex: '840 / 1000')",
            "analiseGeral": "Um parágrafo resumindo a qualidade geral do texto.",
            "c1": "Análise detalhada da Competência 1 (Norma Culta).",
            "c2": "Análise detalhada da Competência 2 (Compreensão/Tema).",
            "c3": "Análise detalhada da Competência 3 (Argumentação).",
            "c4": "Análise detalhada da Competência 4 (Coesão).",
            "c5": "Análise detalhada da Competência 5 (Proposta de Intervenção).",
            "sugestoes": "Lista de 3 a 5 pontos práticos para melhorar (use quebras de linha \\n)."
        }}
        
        Se o tipo da prova ({tipo}) não usar 5 competências exatas, adapte o conteúdo para cobrir os critérios daquela prova, mas MANTENHA AS CHAVES DO JSON IGUAIS (use c1 a c5 para os critérios principais).
        """

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """
        Tenta extrair e converter o texto da IA em um dicionário Python.
        Remove blocos de código Markdown que a IA pode adicionar.
        """
        try:
            # Remove blocos de código Markdown se existirem (```json ... ```)
            cleaned_text = text.replace("```json", "").replace("```", "").strip()
            
            # Tenta fazer o parse
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            logger.error(f"Falha ao fazer parse do JSON da IA. Texto recebido: {text[:100]}...")
            return None