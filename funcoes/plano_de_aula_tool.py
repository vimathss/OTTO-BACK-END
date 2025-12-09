"""
PlanoDeAulaTool - Ferramenta de geração de planos de aula para o OTTO

Este módulo utiliza o LLM para gerar metodologias, objetivos e avaliações
com base nos parâmetros fornecidos pelo professor.
"""

import logging
import json
import re  # --- IMPORTANTE: Importando Regex para extrair JSON ---
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PlanoDeAulaTool:
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        logger.info("PlanoDeAulaTool inicializado.")

    def gerar_plano(
        self,
        user_id: str,
        disciplina: str,
        turma: str,
        carga_horaria: str,
        tema: str,
        subtemas_bncc: str
    ) -> Dict[str, Any]:
        """
        Gera os detalhes pedagógicos do plano de aula.
        """
        try:
            logger.info(f"Gerando plano de aula para {user_id}: {tema} ({disciplina})")

            # 1. Constrói o Prompt
            prompt = self._build_prompt(disciplina, turma, carga_horaria, tema, subtemas_bncc)
            
            # 2. Chama o LLM
            system_prompt = "Você é um coordenador pedagógico especialista em BNCC e metodologias ativas. Seu objetivo é ajudar professores a estruturar aulas engajadoras e bem planejadas."
            
            llm_response = self.llm_manager.generate_response(
                prompt=prompt,
                context=None,
                system_prompt=system_prompt,
                config={"temperature": 0.4}
            )
            
            if not llm_response["success"]:
                return {"success": False, "error": llm_response.get("error")}
            
            # 3. Processa a resposta (JSON Parsing Robusto)
            content_raw = llm_response["content"]
            parsed_content = self._parse_json_response(content_raw)
            
            if not parsed_content:
                # Se falhar, logamos o que a IA retornou para debug
                logger.error(f"FALHA AO PARSEAR JSON. Conteúdo recebido:\n{content_raw}")
                return {
                    "success": False, 
                    "error": "A IA gerou uma resposta inválida. Tente novamente ou simplifique o tema."
                }

            # 4. Retorna no formato esperado
            return {
                "success": True,
                "content": "Plano gerado com sucesso.",
                "user_id": user_id,
                **parsed_content 
            }

        except Exception as e:
            logger.error(f"Erro crítico na PlanoDeAulaTool: {e}")
            return {"success": False, "error": f"Erro interno no servidor: {str(e)}"}

    def _build_prompt(self, disciplina, turma, carga_horaria, tema, bncc) -> str:
        """
        Monta o prompt exigindo saída JSON estrita e divisão por ETAPAS.
        """
        return f"""
        Crie um plano de aula detalhado com as seguintes características:
        
        DADOS DA AULA:
        - Disciplina: {disciplina}
        - Turma/Ano: {turma}
        - Duração: {carga_horaria}
        - Tema Principal: {tema}
        - Habilidades/Códigos da BNCC: {bncc}
        
        ---
        
        INSTRUÇÕES DE SAÍDA (OBRIGATÓRIO):
        Retorne APENAS um objeto JSON válido.
        A estrutura deve ser EXATAMENTE esta:
        
        {{
            "objetivos": "Lista de 3 a 4 objetivos de aprendizagem claros (começando com verbos no infinitivo).",
            "metodologia": "Roteiro dividido em 'Etapas' numeradas (ex: 'Etapa 1: Introdução...', 'Etapa 2: Desenvolvimento...'). Para cada etapa, descreva a atividade. NÃO use marcações de tempo rígidas nos títulos, apenas a sequência lógica.",
            "recursos": "Lista de materiais e recursos necessários.",
            "instrumentosAvaliacao": "Como o professor vai verificar o aprendizado.",
            "criteriosAvaliacao": "Quais critérios específicos serão observados."
        }}
        
        Use formatação Markdown no texto dos valores (ex: **negrito** para destaque, - para listas).
        """

    def _parse_json_response(self, text: str) -> Dict:
        """
        Tenta extrair JSON de forma robusta usando Regex.
        """
        try:
            # 1. Tenta encontrar o JSON delimitado por chaves {}
            match = re.search(r'\{[\s\S]*\}', text)
            
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            
            # 2. Fallback simples
            cleaned_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
            
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.error(f"Erro inesperado no parse JSON: {e}")
            return None