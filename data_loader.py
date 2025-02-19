import os
from typing import List, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import numpy as np
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

@dataclass
class TaggedContent:
    content: str
    source_tags: Set[str]
    ai_tags: Set[str]
    metadata: Dict
    embedding_id: str

class TagNetwork:
    def __init__(self):
        self.graph = nx.Graph()
        self.tag_frequencies = defaultdict(int)
    
    def add_tag_relationship(self, tag1: str, tag2: str, weight: float):
        if tag1 != tag2:
            if self.graph.has_edge(tag1, tag2):
                current_weight = self.graph[tag1][tag2]['weight']
                new_weight = (current_weight + weight) / 2
                self.graph[tag1][tag2]['weight'] = new_weight
            else:
                self.graph.add_edge(tag1, tag2, weight=weight)
        self.tag_frequencies[tag1] += 1
        self.tag_frequencies[tag2] += 1
    
    def get_similar_tag(self, new_tag: str, threshold: float = 0.7) -> str:
        """Return existing similar tag or new_tag if no similar tag exists."""
        for existing_tag in self.graph.nodes():
            if len(set(new_tag).intersection(set(existing_tag))) / max(len(new_tag), len(existing_tag)) > threshold:
                if self.tag_frequencies[existing_tag] > self.tag_frequencies[new_tag]:
                    return existing_tag
        return new_tag

class EnhancedDocumentProcessor:
    def __init__(self, data_dir: str, db_dir: str, user_tags: List[str]):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.stemmer = PorterStemmer()
        self.user_tags = [self.stemmer.stem(tag.strip().lower()) for tag in user_tags]
        self.embeddings = OllamaEmbeddings(
            model="olmo2",
            base_url="http://localhost:11434"
        )

        self.llm = OllamaLLM(
            model="olmo2",
            base_url="http://localhost:11434",
            temperature=0.1
        )
        
        self.tag_network = TagNetwork()
        self.content_cache = {}
        
        self.supported_files = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.html': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader
        }
        
        os.makedirs(os.path.join(db_dir, 'tags'), exist_ok=True)
    
    def generate_ai_tags(self, content: str) -> Set[str]:
        """Generate up to 5 single-word AI tags for content."""
        prompt = f"""
        Analyze this content and generate up to 5 single-word tags.
        Use only fundamental concepts and specific topics.
        Return only the tags separated by commas.
        Content: {content[:500]}...
        """
        
        try:
            response = self.llm.invoke(prompt)
            ai_tags = set()
            for tag in response.split(',')[:5]:
                tag = self.stemmer.stem(tag.strip().lower())
                tag = self.tag_network.get_similar_tag(tag)
                ai_tags.add(tag)
            return ai_tags
        except Exception as e:
            print(f"Error generating AI tags: {str(e)}")
            return set()
    
    def calculate_tag_relationship(self, tag1: str, tag2: str, 
                                 content_vectors: List[np.ndarray]) -> float:
        """Calculate relationship strength between tags based on content similarity."""
        if not content_vectors:
            return 0.0
            
        vectors1 = []
        vectors2 = []
        
        for vec in content_vectors:
            vec_id = id(vec)
            if vec_id in self.content_cache:
                if tag1 in self.content_cache[vec_id].source_tags or tag1 in self.content_cache[vec_id].ai_tags:
                    vectors1.append(vec)
                if tag2 in self.content_cache[vec_id].source_tags or tag2 in self.content_cache[vec_id].ai_tags:
                    vectors2.append(vec)
        
        if not vectors1 or not vectors2:
            return 0.0
            
        vectors1 = np.array(vectors1)
        vectors2 = np.array(vectors2)
        
        norms1 = np.linalg.norm(vectors1, axis=1)
        norms2 = np.linalg.norm(vectors2, axis=1)
        
        similarities = np.dot(vectors1, vectors2.T) / np.outer(norms1, norms2)
        return float(np.mean(similarities))
    
    def process_documents(self) -> List[TaggedContent]:
        """Process all documents in the data directory with progress tracking."""
        all_tagged_contents = []
        
        files_to_process = [
            f for f in os.listdir(self.data_dir)
            if os.path.splitext(f)[1].lower() in self.supported_files
        ]
        
        for file in tqdm(files_to_process, desc="Processing documents"):
            file_path = os.path.join(self.data_dir, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            try:
                loader = self.supported_files[file_extension](file_path)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    add_start_index=True,
                )
                
                for doc in tqdm(documents, desc=f"Processing {file}", leave=False):
                    chunks = text_splitter.split_text(doc.page_content)
                    
                    for chunk in tqdm(chunks, desc="Processing chunks", leave=False):
                        ai_tags = self.generate_ai_tags(chunk)
                        embedding = self.embeddings.embed_query(chunk)
                        
                        tagged_content = TaggedContent(
                            content=chunk,
                            source_tags=set(self.user_tags),
                            ai_tags=ai_tags,
                            metadata={ 'source': file, 'start_index': doc.metadata.get('start_index', 0) },
                            embedding_id=str(hash(chunk))
                        )
                        
                        self.content_cache[id(embedding)] = tagged_content
                        all_tagged_contents.append(tagged_content)
                        
                        all_tags = tagged_content.source_tags.union(tagged_content.ai_tags)
                        for tag1 in all_tags:
                            for tag2 in all_tags:
                                if tag1 < tag2:
                                    weight = self.calculate_tag_relationship(tag1, tag2, [embedding])
                                    self.tag_network.add_tag_relationship(tag1, tag2, weight)
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        return all_tagged_contents

    def create_vector_stores(self, tagged_contents: List[TaggedContent]):
        """Create vector stores for processed content with progress tracking."""
        print("Creating vector stores...")
        
        documents = [
            Document(page_content=tc.content, metadata=tc.metadata)
            for tc in tagged_contents
        ]
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.db_dir, 'main')
        )
        
        all_tags = set()
        for tc in tagged_contents:
            all_tags.update(tc.source_tags)
            all_tags.update(tc.ai_tags)
        
        for tag in tqdm(all_tags, desc="Creating tag-specific vector stores"):
            tag_documents = [
                Document(page_content=tc.content, metadata=tc.metadata)
                for tc in tagged_contents
                if tag in tc.source_tags or tag in tc.ai_tags
            ]
            
            tag_dir = os.path.join(self.db_dir, 'tags', tag)
            os.makedirs(tag_dir, exist_ok=True)
            
            Chroma.from_documents(
                documents=tag_documents,
                embedding=self.embeddings,
                persist_directory=tag_dir
            )

import sys

def main():
    print("Welcome to the Enhanced Document Processor!")
    
    data_dir = input("Enter the path to the data directory: ").strip()
    db_dir = input("Enter the path to the database directory: ").strip()
    user_tags = input("Enter user-defined tags (comma-separated): ").strip().split(',')
    
    print("\nPlease confirm the following details:")
    print(f"Data Directory: {data_dir}")
    print(f"Database Directory: {db_dir}")
    print(f"User Tags: {', '.join(user_tags)}")
    confirm = input("Proceed with these settings? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Operation cancelled.")
        sys.exit(0)
    
    processor = EnhancedDocumentProcessor(data_dir, db_dir, user_tags)
    
    print("\nStarting document processing...")
    tagged_contents = processor.process_documents()
    
    if not tagged_contents:
        print("No documents processed. Exiting.")
        sys.exit(0)
    
    print("\nCreating vector stores...")
    processor.create_vector_stores(tagged_contents)
    
    print("\nProcessing complete! All documents have been indexed successfully.")
    
if __name__ == "__main__":
    main()
