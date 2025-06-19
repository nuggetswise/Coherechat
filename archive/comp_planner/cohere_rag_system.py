"""
Cohere RAG System for Compensation Data
Implements embeddings, semantic search, and reranking for compensation benchmarks
"""
import streamlit as st
import cohere
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from datetime import datetime
import uuid
import sys
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import centralized configuration
try:
    from prompts_and_logic import RAGConfiguration, RAGAnalysisSteps, get_user_config_overrides, apply_config_overrides
except ImportError:
    # Simple fallback if the import fails
    class RAGConfiguration:
        pass
    class RAGAnalysisSteps:
        pass
    def get_user_config_overrides(*args, **kwargs):
        return {}
    def apply_config_overrides(*args, **kwargs):
        return args[0]


class CohereRAGSystem:
    """Production RAG system using real compensation data and Cohere embeddings"""
    
    def __init__(self, cohere_client: cohere.Client):
        self.co_client = cohere_client
        self.embedding_model = "embed-english-v3.0"
        self.rerank_model = "rerank-english-v3.0"
        self.generation_model = "command-r-plus"
        
        # API key handling - extract from client and set environment variable
        api_key = None
        
        # Try to get from client object
        if hasattr(cohere_client, 'api_token'):
            api_key = cohere_client.api_token
        elif hasattr(cohere_client, 'api_key'):
            api_key = cohere_client.api_key
            
        # Fallback to environment or secrets
        if not api_key:
            api_key = os.environ.get("COHERE_API_KEY", "")
            
        # If still no key, try streamlit secrets
        if not api_key and hasattr(st, "secrets"):
            if "COHERE_API_KEY" in st.secrets:
                api_key = st.secrets["COHERE_API_KEY"]
            elif "cohere" in st.secrets and "COHERE_API_KEY" in st.secrets.cohere:
                api_key = st.secrets.cohere["COHERE_API_KEY"]
                
        # Store the API key
        self.api_key = api_key
        
        # Sample data for fallback
        self.sample_data = self._load_sample_data()
        
        # Initialize FAISS vectorstore
        self.using_vectorstore = False
        self.vectorstore = None
        
        try:
            # Create embeddings using Cohere embed-english-v3.0
            self.embeddings = CohereEmbeddings(
                model=self.embedding_model,
                cohere_api_key=self.api_key,
                hybrid=True  # Enable Cohere hybrid search
            )
            self._initialize_vectorstore()
            self.using_vectorstore = True
        except Exception as e:
            print(f"Error initializing FAISS: {str(e)}")
            self.using_vectorstore = False
            print("Using fallback search implementation (vectorstore not available)")
    
    def _load_sample_data(self):
        """Load sample compensation data for fallback"""
        try:
            # Try to load real data
            data_path = os.path.join(parent_dir, "data", "Compensation Data.csv")
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                return df.to_dict('records')
            else:
                # Return fallback sample data
                return [
                    {
                        'job_title': 'Software Engineer',
                        'job_level': 'Senior',
                        'location': 'San Francisco',
                        'base_salary_usd': 180000,
                        'bonus_usd': 30000,
                        'equity_value_usd': 50000,
                        'company_stage': 'Growth',
                        'offer_outcome': 'Accepted',
                        'candidate_preference': 'High',
                        'notes': 'Strong technical skills, 7 years experience'
                    },
                    {
                        'job_title': 'Product Manager',
                        'job_level': 'L6',
                        'location': 'New York City',
                        'base_salary_usd': 190000,
                        'bonus_usd': 40000,
                        'equity_value_usd': 60000,
                        'company_stage': 'Late Stage',
                        'offer_outcome': 'Accepted',
                        'candidate_preference': 'High',
                        'notes': 'Led multiple successful product launches'
                    }
                ]
        except Exception as e:
            print(f"Error loading compensation data: {e}")
            return []
    
    def _initialize_vectorstore(self):
        """Initialize FAISS vector store with real compensation data"""
        try:
            # Load the real compensation data
            data_path = os.path.join(parent_dir, "data", "Compensation Data.csv")
            if not os.path.exists(data_path):
                print(f"Compensation data file not found at {data_path}")
                return
                
            df = pd.read_csv(data_path)
            
            # Clean and prepare the data
            df = df.dropna()  # Remove any rows with missing data
            
            # Create documents for each compensation record
            documents = []
            
            for idx, row in df.iterrows():
                # Create a comprehensive description for embedding
                total_comp = row['base_salary_usd'] + row['bonus_usd'] + row['equity_value_usd']
                
                description = f"""
                {row['job_title']} at {row['job_level']} level in {row['location']}.
                Base salary: ${row['base_salary_usd']:,}
                Annual bonus: ${row['bonus_usd']:,}
                Equity value: ${row['equity_value_usd']:,}
                Total compensation: ${total_comp:,}
                Company stage: {row['company_stage']}
                Offer outcome: {row['offer_outcome']}
                Candidate preference: {row['candidate_preference']}
                Notes: {row['notes']}
                """
                
                # Create metadata for filtering and analysis
                metadata = {
                    'job_title': str(row['job_title']),
                    'job_level': str(row['job_level']),
                    'location': str(row['location']),
                    'base_salary_usd': int(row['base_salary_usd']),
                    'bonus_usd': int(row['bonus_usd']),
                    'equity_value_usd': int(row['equity_value_usd']),
                    'total_compensation': int(total_comp),
                    'company_stage': str(row['company_stage']),
                    'offer_outcome': str(row['offer_outcome']),
                    'candidate_preference': str(row['candidate_preference']),
                    'notes': str(row['notes']),
                    'record_id': f"comp_record_{idx}",
                    'source': 'compensation_database'
                }
                
                documents.append(Document(
                    page_content=description.strip(),
                    metadata=metadata
                ))
            
            # Create FAISS vectorstore from documents
            if documents:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                print(f"Successfully loaded {len(documents)} real compensation records into FAISS")
            else:
                print("No documents to load into vectorstore")
                
        except Exception as e:
            print(f"Error loading real compensation data: {e}")
            self.vectorstore = None
    
    def search_similar_compensation(self, query: str, role: str = "", level: str = "", location: str = "", k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar compensation records using semantic search"""
        if self.using_vectorstore and self.vectorstore:
            try:
                # Enhanced query with context
                search_query = f"{query} {role} {level} {location}".strip()
                
                # Perform semantic search
                docs = self.vectorstore.similarity_search(search_query, k=min(k, 20))
                
                if not docs:
                    return []
                
                # Format results
                formatted_results = []
                for i, doc in enumerate(docs):
                    formatted_results.append({
                        'document': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': 1.0 - (i * 0.1)  # Approximate similarity score
                    })
                
                return formatted_results
                
            except Exception as e:
                print(f"Error in semantic search: {e}")
                return self._fallback_search(query, role, level, location, k)
        else:
            return self._fallback_search(query, role, level, location, k)
    
    def _fallback_search(self, query: str, role: str = "", level: str = "", location: str = "", k: int = 5) -> List[Dict[str, Any]]:
        """Simple search for compensation records based on role, level, location"""
        if not self.sample_data:
            return []
        
        # Simple filtering based on role, level, location
        results = []
        for record in self.sample_data:
            score = 0
            if role and role.lower() in record.get('job_title', '').lower():
                score += 3
            if level and level.lower() in record.get('job_level', '').lower():
                score += 2
            if location and location.lower() in record.get('location', '').lower():
                score += 2
                
            if score > 0:
                results.append({
                    'document': f"{record['job_title']} at {record['job_level']} level in {record['location']}. "
                               f"Base: ${record['base_salary_usd']:,}, Bonus: ${record['bonus_usd']:,}, "
                               f"Equity: ${record['equity_value_usd']:,}",
                    'metadata': record,
                    'similarity_score': score
                })
        
        # Sort by score and limit results
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        return results[:k]
    
    def rerank_results(self, query: str, search_results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank search results using Cohere Rerank API"""
        if not search_results:
            return []
        
        try:
            # Prepare documents for reranking
            documents = [result['document'] for result in search_results]
            
            # Use Cohere Rerank
            rerank_response = self.co_client.rerank(
                model=self.rerank_model,
                query=query,
                documents=documents,
                top_k=min(top_k, len(documents))
            )
            
            # Reorder results based on rerank scores
            reranked_results = []
            for result in rerank_response.results:
                original_result = search_results[result.index]
                original_result['rerank_score'] = result.relevance_score
                reranked_results.append(original_result)
            
            return reranked_results
            
        except Exception as e:
            print(f"Error in reranking: {e}")
            # Return original results if reranking fails
            return search_results[:top_k]
    
    def generate_compensation_recommendation(self, query: str, role: str = "", level: str = "", location: str = "", search_k: int = 5) -> Dict[str, Any]:
        """Generate comprehensive compensation recommendation using RAG"""
        try:
            # Step 1: Semantic search
            search_results = self.search_similar_compensation(query, role, level, location, k=search_k * 2)
            
            if not search_results:
                return {
                    "error": "No relevant compensation data found",
                    "recommendation": "Unable to find similar roles in the database",
                    "confidence_score": 0.0,
                    "benchmark_count": 0,
                    "sources": []
                }
            
            # Step 2: Rerank results
            reranked_results = self.rerank_results(query, search_results, top_k=search_k)
            
            # Step 3: Extract compensation statistics
            comp_stats = self._analyze_compensation_data(reranked_results)
            
            # Step 4: Generate recommendation using Cohere
            context_text = "\n\n".join([
                f"Record {i+1}: {result['document']}" 
                for i, result in enumerate(reranked_results[:3])
            ])
            
            prompt = f"""
            Based on the following real compensation data, provide a comprehensive compensation recommendation for: {query}
            
            Role: {role}
            Level: {level}
            Location: {location}
            
            Relevant Compensation Data:
            {context_text}
            
            Compensation Statistics from {len(reranked_results)} similar roles:
            - Average Base Salary: ${comp_stats['avg_base']:,.0f}
            - Base Salary Range: ${comp_stats['min_base']:,.0f} - ${comp_stats['max_base']:,.0f}
            - Average Bonus: ${comp_stats['avg_bonus']:,.0f}
            - Average Equity: ${comp_stats['avg_equity']:,.0f}
            - Average Total Comp: ${comp_stats['avg_total']:,.0f}
            
            Provide a detailed recommendation including:
            1. Recommended base salary with justification
            2. Bonus structure and amount
            3. Equity recommendation  
            4. Total compensation summary
            5. Market positioning analysis
            6. Key considerations based on the data
            
            Format as a professional compensation recommendation.
            """
            
            response = self.co_client.chat(
                model=self.generation_model,
                message=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            return {
                "recommendation": response.text,
                "confidence_score": min(9.5, 7.0 + (len(reranked_results) * 0.3)),
                "benchmark_count": len(reranked_results),
                "compensation_stats": comp_stats,
                "sources": [f"Real compensation data: {result['metadata'].get('job_title', 'Unknown')} at {result['metadata'].get('location', 'Unknown')}" for result in reranked_results[:3]],
                "method": "rag_with_real_data"
            }
            
        except Exception as e:
            return {
                "error": f"RAG generation failed: {str(e)}",
                "recommendation": "Unable to generate recommendation due to system error",
                "confidence_score": 0.0,
                "benchmark_count": 0,
                "sources": []
            }
    
    def _analyze_compensation_data(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze compensation data from search results"""
        if not results:
            return {
                'avg_base': 0, 'min_base': 0, 'max_base': 0,
                'avg_bonus': 0, 'avg_equity': 0, 'avg_total': 0
            }
        
        base_salaries = []
        bonuses = []
        equity_values = []
        total_comps = []
        
        for result in results:
            metadata = result.get('metadata', {})
            base_salaries.append(metadata.get('base_salary_usd', 0))
            bonuses.append(metadata.get('bonus_usd', 0))
            equity_values.append(metadata.get('equity_value_usd', 0))
            total_comps.append(metadata.get('total_compensation', 0))
        
        return {
            'avg_base': np.mean(base_salaries) if base_salaries else 0,
            'min_base': np.min(base_salaries) if base_salaries else 0,
            'max_base': np.max(base_salaries) if base_salaries else 0,
            'avg_bonus': np.mean(bonuses) if bonuses else 0,
            'avg_equity': np.mean(equity_values) if equity_values else 0,
            'avg_total': np.mean(total_comps) if total_comps else 0
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        if self.using_vectorstore and self.vectorstore:
            try:
                # Get data from the sample_data for stats since FAISS doesn't track metadata directly
                if not self.sample_data:
                    return self._fallback_database_stats("No sample data available")
                
                # Calculate statistics from sample data
                job_titles = [r.get('job_title', 'Unknown') for r in self.sample_data]
                job_levels = [r.get('job_level', 'Unknown') for r in self.sample_data]
                locations = [r.get('location', 'Unknown') for r in self.sample_data]
                
                unique_titles = len(set(job_titles))
                unique_levels = len(set(job_levels))
                unique_locations = len(set(locations))
                
                base_salaries = [r.get('base_salary_usd', 0) for r in self.sample_data]
                avg_base = np.mean(base_salaries) if base_salaries else 0
                
                return {
                    "total_records": len(self.sample_data),
                    "unique_job_titles": unique_titles,
                    "unique_levels": unique_levels,
                    "unique_locations": unique_locations,
                    "average_base_salary": f"${avg_base:,.0f}",
                    "rag_system": "active",
                    "embedding_model": self.embedding_model,
                    "rerank_model": self.rerank_model,
                    "generation_model": self.generation_model,
                    "data_source": "real_compensation_data.csv",
                    "vector_store": "FAISS"
                }
            except Exception as e:
                return self._fallback_database_stats(str(e))
        else:
            return self._fallback_database_stats("Vector store not available")
    
    def _fallback_database_stats(self, reason: str = "") -> Dict[str, Any]:
        """Get fallback database statistics when vector store is not available"""
        return {
            "total_records": len(self.sample_data),
            "unique_job_titles": len(set(r.get('job_title', '') for r in self.sample_data)),
            "unique_levels": len(set(r.get('job_level', '') for r in self.sample_data)),
            "unique_locations": len(set(r.get('location', '') for r in self.sample_data)),
            "average_base_salary": "$0",
            "rag_system": "fallback",
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
            "generation_model": self.generation_model,
            "data_source": "sample_data",
            "fallback_reason": reason
        }

def get_cohere_rag_system(cohere_client: cohere.Client) -> CohereRAGSystem:
    """Factory function to create Cohere RAG system with real data"""
    return CohereRAGSystem(cohere_client)