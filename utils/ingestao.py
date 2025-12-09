"""
Script de Ingestão - Processamento e indexação de documentos para o OTTO

Este script carrega, processa e indexa documentos para uso pelo assistente OTTO,
criando ou atualizando vector stores para diferentes tipos de conteúdo.

Comando para execução:

python utils/ingestao.py --data-dir data/NOME --store-name NOME_VECTOR_knowledge
python utils/ingestao.py --data-dir data/main --store-name main 

"""
import os
import argparse
import logging
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingestao.log"),
        logging.StreamHandler()
    ]
)

# Adicionar diretório pai ao path para importar módulos do OTTO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_store import VectorStoreManager

def main():
    """Script para ingestão de documentos."""
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Ingestão de Documentos para OTTO')
    parser.add_argument('--data-dir', type=str, required=True, help='Diretório de dados')
    parser.add_argument('--store-name', type=str, default='main', help='Nome do vector store')
    args = parser.parse_args()
    
    try:
        # Verificar se o diretório de dados existe
        if not os.path.exists(args.data_dir):
            logging.error(f"Diretório de dados '{args.data_dir}' não existe")
            return
        
        # Inicializar vector store manager
        # Caminho absoluto para a pasta "vector_stores" na raiz do projeto
        otto_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        base_dir = os.path.join(otto_root_dir, "vector_stores")

        vector_store_manager = VectorStoreManager(base_dir=base_dir)

        
        # Criar ou atualizar vector store
        logging.info(f"Ingerindo documentos de '{args.data_dir}' para vector store '{args.store_name}'...")
        
        # O create_or_load agora lida com a lógica de persistência/criação
        vector_store_manager.create_or_load(args.store_name, args.data_dir)
        
        logging.info("Ingestão concluída com sucesso!")
        
    except Exception as e:
        logging.error(f"Erro durante ingestão: {e}")
        print(f"Erro durante ingestão: {e}")

if __name__ == "__main__":
    main()
