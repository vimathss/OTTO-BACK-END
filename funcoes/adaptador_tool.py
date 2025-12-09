"""
AdaptadorTool - Ferramenta para adaptar textos pedagógicos

Permite simplificar linguagem, resumir, criar glossários ou adaptar
para necessidades específicas (ex: dislexia, TDAH).
"""

import logging
import json
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AdaptadorTool:
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        logger.info("AdaptadorTool inicializado.")

    def adaptar_conteudo(
        self,
        user_id: str,
        texto_original: str,
        tipo_adaptacao: str, # ex: 'simplificar', 'resumir', 'glossario'
        comentarios_extras: str = ""
    ) -> Dict[str, Any]:
        """
        Adapta o texto fornecido conforme o objetivo selecionado.
        """
        try:
            logger.info(f"Adaptando texto para {user_id}. Tipo: {tipo_adaptacao}")

            # 1. Seleciona a instrução baseada no tipo
            instrucao_especifica = self._get_instruction(tipo_adaptacao)
            
            # 2. Constrói o Prompt
            prompt = f"""
            Você é um especialista em didática e inclusão escolar.
            
            TAREFA:
            {instrucao_especifica}
            
            OBSERVAÇÕES DO PROFESSOR:
            "{comentarios_extras}"
            
            TEXTO ORIGINAL:
            \"\"\"
            {texto_original}
            \"\"\"
            
            ---
            INSTRUÇÕES DE SAÍDA:
            Retorne APENAS um objeto JSON válido com a seguinte estrutura:
            {{
                "titulo": "Um título sugerido para o material adaptado",
                "conteudoAdaptado": "O texto completo já transformado/adaptado (use Markdown para formatar títulos, negritos e listas)."
            }}
            """
            
            # 3. Chama o LLM
            llm_response = self.llm_manager.generate_response(
                prompt=prompt,
                context=None,
                config={"temperature": 0.3} # Baixa criatividade para manter fidelidade ao texto original
            )
            
            if not llm_response["success"]:
                return {"success": False, "error": llm_response.get("error")}
            
            # 4. Processa a resposta
            content_raw = llm_response["content"]
            parsed_content = self._parse_json_response(content_raw)
            
            if not parsed_content:
                return {"success": False, "error": "Falha ao estruturar a adaptação."}

            return {
                "success": True,
                "user_id": user_id,
                **parsed_content
            }

        except Exception as e:
            logger.error(f"Erro na AdaptadorTool: {e}")
            return {"success": False, "error": str(e)}

    def _get_instruction(self, tipo: str) -> str:
        """Retorna a instrução de prompt adequada para cada tipo."""
        instrucoes = {
            "simplificar": "Reescreva o texto simplificando o vocabulário e a estrutura das frases para torná-lo acessível a alunos do Ensino Fundamental I ou com dificuldades de leitura. Mantenha as ideias centrais, mas use linguagem direta.",
            "resumir": "Crie um resumo esquemático do texto. Identifique os conceitos-chave e organize-os em tópicos (bullet points) claros e hierárquicos para facilitar a revisão.",
            "glossario": "Analise o texto, identifique palavras complexas, termos técnicos ou conceitos difíceis e crie um GLOSSÁRIO. Liste a palavra e sua definição simplificada.",
            "dislexia": "Adapte o texto para alunos com dislexia. Use frases curtas e ordem direta (sujeito-verbo-objeto). Evite metáforas complexas. Destaque em NEGRITO as palavras-chave de cada parágrafo.",
            "mapa_mental": "Transforme o texto em uma estrutura de Mapa Mental em texto (tópicos indentados). Comece pelo tema central e vá ramificando os subtemas e detalhes."
        }
        return instrucoes.get(tipo, "Adapte o texto conforme solicitado pelo professor, melhorando a clareza.")

    def _parse_json_response(self, text: str) -> Dict:
        try:
            match = re.search(r'\{[\s\S]*\}', text)
            if match: return json.loads(match.group(0))
            return json.loads(text.replace("```json", "").replace("```", "").strip())
        except:
            return None