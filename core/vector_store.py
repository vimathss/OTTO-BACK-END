import os
import shutil
import logging
from typing import List, Any, Optional, Dict
from langchain_core.documents import Document

# --- MUDANÇA PRINCIPAL: Usando a nova lib dedicada ---
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Gerencia diferentes vector stores para diferentes conjuntos de dados."""

    def __init__(self, base_dir: str = "./vector_stores"):
        self.base_dir = base_dir
        self.stores: dict[str, Any] = {}
        self.embeddings = None

        os.makedirs(base_dir, exist_ok=True)
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """Inicializa o modelo de embeddings."""
        try:
            # O modelo "all-MiniLM-L6-v2" é leve e roda bem em CPU
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            raise RuntimeError(f"Não foi possível inicializar o modelo de embeddings: {e}")

    def get_store(self, name: str) -> Any:
        """Obtém um vector store pelo nome, carregando-o se necessário."""
        if name not in self.stores:
            self._load_store(name)
        return self.stores.get(name)

    def _load_store(self, name: str) -> None:
        """Carrega um vector store existente do disco."""
        store_dir = os.path.join(self.base_dir, name)
        if os.path.exists(store_dir):
            try:
                # Na nova versão, passamos o diretório e a função de embedding
                self.stores[name] = Chroma(
                    persist_directory=store_dir,
                    embedding_function=self.embeddings,
                    collection_name=name # Boa prática: nomear a coleção
                )
                logger.info(f"Vector store '{name}' carregado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao carregar vector store '{name}': {e}")
                self.stores[name] = None
        else:
            logger.warning(f"Vector store '{name}' não existe no caminho: {store_dir}")
            self.stores[name] = None

    def create_or_load(self, name: str, documents_dir: Optional[str] = None) -> Any:
        """Cria um novo vector store ou carrega um existente."""
        store_dir = os.path.join(self.base_dir, name)

        # Se já existe e não estamos forçando recriação (lógica simples), carregamos
        # Nota: Se quiser forçar recriação na ingestão, delete a pasta antes de chamar isso.
        if os.path.exists(store_dir) and not documents_dir:
             store = self.get_store(name)
             if store:
                 return store

        if not documents_dir:
            # Se não existe e não tem documentos, não dá pra criar
            if not os.path.exists(store_dir):
                 raise ValueError(f"Vector store '{name}' não existe e nenhum diretório foi fornecido")
            return self.get_store(name)

        # Se forneceu diretório, vamos processar os documentos
        documents = self._load_documents(documents_dir)
        if not documents:
            raise ValueError(f"Nenhum documento encontrado em '{documents_dir}'")

        try:
            # Se a pasta já existe, vamos limpar para garantir uma recriação limpa
            # Isso resolve problemas de conflito de IDs ou dados antigos
            if os.path.exists(store_dir):
                logger.info(f"Removendo store antigo em {store_dir} para recriação limpa...")
                shutil.rmtree(store_dir)

            # Criar nova store
            self.stores[name] = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings, # Nota: O nome do parametro mudou em algumas versões, mas embedding_function costuma funcionar. Na langchain-chroma usa-se 'embedding' ou 'embedding_function' dependendo da versão exata, mas o construtor padrão acima é mais seguro.
                persist_directory=store_dir,
                collection_name=name
            )
            logger.info(f"Vector store '{name}' criado com sucesso com {len(documents)} documentos")
            return self.stores[name]
        except Exception as e:
            raise RuntimeError(f"Erro ao criar vector store '{name}': {e}")

    def _load_documents(self, documents_dir: str) -> List[Document]:
        """Carrega e divide documentos de um diretório."""
        documents: List[Document] = []

        if not os.path.exists(documents_dir):
            logger.error(f"Diretório '{documents_dir}' não existe")
            return documents

        # Carregadores
        loaders = [
            ("PDF", DirectoryLoader(documents_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)),
            ("DOCX", DirectoryLoader(documents_dir, glob="**/*.docx", loader_cls=Docx2txtLoader)),
            ("TXT", DirectoryLoader(documents_dir, glob="**/*.txt", loader_cls=TextLoader))
        ]

        for tipo, loader in loaders:
            try:
                loaded = loader.load()
                documents.extend(loaded)
                logger.info(f"Carregados {len(loaded)} documentos {tipo}")
            except Exception as e:
                logger.warning(f"Erro ao carregar {tipo}s: {e}")

        # CSV Manual
        import glob
        try:
            for csv_file in glob.glob(os.path.join(documents_dir, "**/*.csv"), recursive=True):
                loader = CSVLoader(file_path=csv_file)
                documents.extend(loader.load())
            logger.info(f"Documentos CSV processados.")
        except Exception as e:
            logger.warning(f"Erro ao carregar CSVs: {e}")

        # JSON Manual
        try:
            import json
            for json_file in glob.glob(os.path.join(documents_dir, "**/*.json"), recursive=True):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Lógica simplificada para JSON
                content = json.dumps(data, ensure_ascii=False)
                documents.append(Document(page_content=content, metadata={"source": json_file}))
            logger.info(f"Documentos JSON processados.")
        except Exception as e:
            logger.warning(f"Erro ao carregar JSONs: {e}")

        # Dividir documentos
        if documents:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            )
            documents = splitter.split_documents(documents)
            logger.info(f"Documentos divididos em {len(documents)} chunks")

        return documents

    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Busca semântica em uma collection."""
        try:
            store = self.get_store(collection_name)
            if store is None:
                return {"success": False, "error": f"Collection '{collection_name}' não encontrada"}

            results_with_scores = store.similarity_search_with_score(
                query=query,
                k=n_results,
                filter=filter_metadata
            )

            formatted = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance": score
                }
                for doc, score in results_with_scores
            ]

            return {"success": True, "results": formatted}

        except Exception as e:
            logger.error(f"Erro na busca semântica: {e}")
            return {"success": False, "error": str(e)}