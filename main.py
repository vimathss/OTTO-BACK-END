#uvicorn main:app --reload

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar componentes do OTTO
from core.llm_manager import LLMManager
from core.vector_store import VectorStoreManager
from core.memory_manager import FirebaseMemoryManager as MemoryManager
from core.agent import OTTOAgent
from funcoes.redacao_tool import RedacaoTool
from funcoes.plano_de_aula_tool import PlanoDeAulaTool 
from funcoes.adaptador_tool import AdaptadorTool 


# --- MODELOS DE ENTRADA (Payloads) ---

class ChatIn(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[str] = None
    conversation_date: Optional[str] = None

class RedacaoIn(BaseModel):
    user_id: str
    tipo: str
    titulo: str
    alunoNome: str
    alunoSala: str
    redacao: str
    comentarios: Optional[str] = ""

class PlanoDeAulaIn(BaseModel):
    user_id: str
    disciplina: str
    turma: str
    cargaHoraria: str
    tema: str
    subtemasBNCC: str


class AdaptadorIn(BaseModel):
    user_id: str
    textoOriginal: str
    tipoAdaptacao: str
    comentarios: Optional[str] = ""

# --- INICIALIZAÇÃO ---

llm_manager = LLMManager()
vector_store_manager = VectorStoreManager()
memory_manager = MemoryManager()

agent = OTTOAgent(llm_manager, vector_store_manager, memory_manager)
redacao_tool = RedacaoTool(llm_manager, vector_store_manager)
plano_tool = PlanoDeAulaTool(llm_manager) 
adaptador_tool = AdaptadorTool(llm_manager) 


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="OTTO - Assistente de IA para Professores",
    description="API Backend para o assistente educacional OTTO.",
    version="1.0.0"
)

# Configuração do CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global
app.state.agent = agent
app.state.redacao_tool = redacao_tool
app.state.plano_tool = plano_tool 
app.state.adaptador_tool = adaptador_tool 


logger.info("OTTO API inicializada com sucesso.")

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"message": "OTTO API está online!"}

@app.post("/chat", response_model=Dict[str, Any])
async def chat_endpoint(payload: ChatIn, request: Request):
    """Endpoint para o chat conversacional."""
    agent = request.app.state.agent
    if agent is None: raise HTTPException(status_code=503, detail="Serviço OTTO indisponível.")
    
    resposta = agent.process_query(
        user_id=payload.user_id,
        message=payload.message,
        conversation_id=payload.conversation_id,
        conversation_date=payload.conversation_date
    )
    if not resposta["success"]: raise HTTPException(status_code=500, detail=resposta["error"])
    return resposta

@app.post("/redacao", response_model=Dict[str, Any])
async def redacao_endpoint(payload: RedacaoIn, request: Request):
    """Endpoint para correção de redação."""
    redacao_tool = request.app.state.redacao_tool
    if redacao_tool is None: raise HTTPException(status_code=503, detail="Serviço de Redação indisponível.")
        
    resultado = redacao_tool.corrigir_redacao(
        user_id=payload.user_id,
        essay_text=payload.redacao,
        essay_type=payload.tipo,
        essay_title=payload.titulo,
        student_name=payload.alunoNome,
        student_class=payload.alunoSala,
        teacher_comments=payload.comentarios
    )
    if not resultado["success"]: raise HTTPException(status_code=500, detail=resultado.get("error"))
    return resultado

# --- NOVO ENDPOINT ---
@app.post("/plano-de-aula", response_model=Dict[str, Any])
async def plano_de_aula_endpoint(payload: PlanoDeAulaIn, request: Request):
    """Endpoint para geração de plano de aula."""
    plano_tool = request.app.state.plano_tool
    
    if plano_tool is None:
        raise HTTPException(status_code=503, detail="Serviço de Plano de Aula indisponível.")
        
    resultado = plano_tool.gerar_plano(
        user_id=payload.user_id,
        disciplina=payload.disciplina,
        turma=payload.turma,
        carga_horaria=payload.cargaHoraria,
        tema=payload.tema,
        subtemas_bncc=payload.subtemasBNCC
    )
    
    if not resultado["success"]:
        raise HTTPException(status_code=500, detail=resultado.get("error"))
        
    return resultado

# --- NOVO ENDPOINT ---
@app.post("/adaptar", response_model=Dict[str, Any])
async def adaptador_endpoint(payload: AdaptadorIn, request: Request):
    """Endpoint para adaptar conteúdo didático."""
    tool = request.app.state.adaptador_tool
    
    if tool is None:
        raise HTTPException(status_code=503, detail="Serviço de Adaptação indisponível.")
        
    resultado = tool.adaptar_conteudo(
        user_id=payload.user_id,
        texto_original=payload.textoOriginal,
        tipo_adaptacao=payload.tipoAdaptacao,
        comentarios_extras=payload.comentarios
    )
    
    if not resultado["success"]:
        raise HTTPException(status_code=500, detail=resultado.get("error"))
        
    return resultado