"""
IntentDetector - Componente desativado temporariamente.
A lógica de detecção de intenção foi removida para simplificar o fluxo.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class IntentDetector:
    """
    Detector de Intenção desativado.
    Sempre retorna a intenção 'chat_geral'.
    """
    
    def __init__(self):
        logger.warning("IntentDetector inicializado, mas está desativado. Sempre retornará 'chat_geral'.")
        
    def detect_intent(self, message: str, context: Dict[str, Any]) -> Any:
        """
        Simula a detecção de intenção, sempre retornando 'chat_geral'.
        """
        class IntentResult:
            def __init__(self, intent, confidence, matched_patterns):
                self.intent = intent
                self.confidence = confidence
                self.matched_patterns = matched_patterns
                
        return IntentResult(
            intent="chat_geral",
            confidence=1.0,
            matched_patterns=[]
        )

# Manter compatibilidade com o nome original
IntentDetector = IntentDetector
