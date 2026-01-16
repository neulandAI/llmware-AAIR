"""Main LinearRAG implementation.

A graph-based RAG system using Named Entity Recognition and Personalized PageRank
for multi-hop reasoning.
"""

import os
import json
import time
import math
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Set, Optional, Union

import numpy as np

from tqdm import tqdm

from .config import LinearRAGConfig
from .graph_builder import GraphBuilder
from ..ner.spacy_ner import SpacyNER
from ..storage.embedding_store import EmbeddingStore
from ..utils import strip_passage_prefix, min_max_normalize

logger = logging.getLogger(__name__)


class LinearRAG:
    """Graph-based Retrieval-Augmented Generation.
    
    LinearRAG uses Named Entity Recognition to build an entity-passage graph,
    then applies Personalized PageRank for multi-hop retrieval.
    
    Key features:
    - Context-preserving entity extraction via SpaCy NER
    - Graph-based retrieval with Personalized PageRank
    - Dense passage retrieval fallback when no entities found
    - Parallel QA generation with configurable LLM
    
    Args:
        config: LinearRAGConfig with all parameters
        
    Example:
        >>> from llmware.RAGMethods.LinearRAG import LinearRAG, LinearRAGConfig
        >>> from llmware.RAGMethods.LinearRAG.adapters import SentenceTransformerAdapter
        >>> 
        >>> config = LinearRAGConfig(
        ...     dataset_name="my_data",
        ...     embedding_model=SentenceTransformerAdapter("all-mpnet-base-v2"),
        ...     working_dir="./linearrag_data"
        ... )
        >>> rag = LinearRAG(config)
        >>> rag.index(passages)
        >>> results = rag.retrieve(questions)
    """
    
    # Default system prompt for QA
    DEFAULT_SYSTEM_PROMPT = (
        "As an advanced reading comprehension assistant, your task is to analyze "
        "text passages and corresponding questions meticulously. Your response start "
        'after "Thought: ", where you will methodically break down the reasoning '
        'process, illustrating how you arrive at conclusions. Conclude with "Answer: " '
        "to present a concise, definitive response, devoid of additional elaborations."
    )
    
    def __init__(self, config: LinearRAGConfig):
        self.config = config
        logger.info(f"Initializing LinearRAG for dataset: {config.dataset_name}")
        
        self.dataset_name = config.dataset_name
        
        # Initialize embedding stores
        self._init_embedding_stores()
        
        # Initialize NER
        self.spacy_ner = SpacyNER(config.spacy_model)
        
        # Initialize graph builder
        self.graph_builder = GraphBuilder(
            self.passage_embedding_store,
            self.entity_embedding_store
        )
        
        # LLM model for QA (optional)
        self.llm_model = config.llm_model
        
        # Entity-sentence mappings (populated during indexing)
        self.entity_hash_id_to_sentence_hash_ids: Dict[str, List[str]] = {}
        self.sentence_hash_id_to_entity_hash_ids: Dict[str, List[str]] = {}
        self.entity_to_sentence: Dict[str, Set[str]] = {}
        self.sentence_to_entity: Dict[str, Set[str]] = {}
        
        # Cached embeddings for retrieval (populated during retrieve)
        self._entity_hash_ids: List[str] = []
        self._entity_embeddings: Optional[np.ndarray] = None
        self._passage_hash_ids: List[str] = []
        self._passage_embeddings: Optional[np.ndarray] = None
        self._sentence_hash_ids: List[str] = []
        self._sentence_embeddings: Optional[np.ndarray] = None
    
    def _init_embedding_stores(self) -> None:
        """Initialize the three embedding stores for passages, entities, and sentences."""
        base_path = os.path.join(self.config.working_dir, self.dataset_name)
        
        self.passage_embedding_store = EmbeddingStore(
            embedding_model=self.config.embedding_model,
            db_filename=os.path.join(base_path, "passage_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="passage"
        )
        
        self.entity_embedding_store = EmbeddingStore(
            embedding_model=self.config.embedding_model,
            db_filename=os.path.join(base_path, "entity_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="entity"
        )
        
        self.sentence_embedding_store = EmbeddingStore(
            embedding_model=self.config.embedding_model,
            db_filename=os.path.join(base_path, "sentence_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="sentence"
        )
    
    # =========================================================================
    # INDEXING
    # =========================================================================
    
    def index(self, passages: Union[List[str], List[Dict[str, Any]]]) -> None:
        """Index passages by extracting entities and building the graph.
        
        This method:
        1. Stores passage embeddings (with optional metadata)
        2. Extracts entities via NER
        3. Stores entity and sentence embeddings
        4. Builds the entity-passage graph
        
        Args:
            passages: Either a list of passage text strings, or a list of dicts with:
                - "text": The passage text (required)
                - "file_source": Source document filename (optional, for evaluation)
                - "page_num": Page number in source document (optional, for evaluation)
                
        Note:
            Metadata (file_source, page_num) is required for evaluation. Without it,
            the evaluator cannot compare retrieved passages against ground truth.
        """
        logger.info(f"Indexing {len(passages)} passages...")
        
        # Store passage embeddings (handles both str and dict formats)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        
        # Load existing NER results or compute new ones
        existing_entities, existing_sentences, new_hash_ids = \
            self._load_existing_ner_data(hash_id_to_passage.keys())
        
        if new_hash_ids:
            new_hash_id_to_passage = {k: hash_id_to_passage[k] for k in new_hash_ids}
            new_entities, new_sentences = self.spacy_ner.batch_ner(
                new_hash_id_to_passage,
                self.config.max_workers
            )
            self._merge_ner_results(existing_entities, existing_sentences, 
                                   new_entities, new_sentences)
        
        self._save_ner_results(existing_entities, existing_sentences)
        
        # Extract nodes and edges from NER results
        entity_nodes, sentence_nodes, passage_to_entities, \
            self.entity_to_sentence, self.sentence_to_entity = \
            self._extract_nodes_and_edges(existing_entities, existing_sentences)
        
        # Store entity and sentence embeddings
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        
        # Build entity-sentence mappings
        self._build_entity_sentence_mappings()
        
        # Build the graph
        self.graph_builder.build_graph(passage_to_entities)
        
        # Save graph
        graph_path = os.path.join(
            self.config.working_dir, 
            self.dataset_name, 
            "LinearRAG.graphml"
        )
        self.graph_builder.save_graph(graph_path)
        
        logger.info(f"Indexing complete. Graph saved to {graph_path}")
    
    def _load_existing_ner_data(
        self, 
        passage_hash_ids
    ) -> Tuple[Dict, Dict, Set]:
        """Load existing NER results from disk if available."""
        ner_path = os.path.join(
            self.config.working_dir, 
            self.dataset_name, 
            "ner_results.json"
        )
        self._ner_results_path = ner_path
        
        if os.path.exists(ner_path):
            with open(ner_path, 'r') as f:
                existing = json.load(f)
            existing_entities = existing.get("passage_hash_id_to_entities", {})
            existing_sentences = existing.get("sentence_to_entities", {})
            existing_hash_ids = set(existing_entities.keys())
            new_hash_ids = set(passage_hash_ids) - existing_hash_ids
            return existing_entities, existing_sentences, new_hash_ids
        
        return {}, {}, set(passage_hash_ids)
    
    def _merge_ner_results(
        self,
        existing_entities: Dict,
        existing_sentences: Dict,
        new_entities: Dict,
        new_sentences: Dict
    ) -> None:
        """Merge new NER results with existing ones."""
        existing_entities.update(new_entities)
        existing_sentences.update(new_sentences)
    
    def _save_ner_results(
        self,
        entities: Dict,
        sentences: Dict
    ) -> None:
        """Save NER results to disk."""
        os.makedirs(os.path.dirname(self._ner_results_path), exist_ok=True)
        with open(self._ner_results_path, 'w') as f:
            json.dump({
                "passage_hash_id_to_entities": entities,
                "sentence_to_entities": sentences
            }, f)
    
    def _extract_nodes_and_edges(
        self,
        passage_entities: Dict[str, List[str]],
        sentence_entities: Dict[str, List[str]]
    ) -> Tuple[Set[str], Set[str], Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Extract entity and sentence nodes, and their relationships."""
        entity_nodes: Set[str] = set()
        sentence_nodes: Set[str] = set()
        passage_hash_id_to_entities: Dict[str, Set[str]] = defaultdict(set)
        entity_to_sentence: Dict[str, Set[str]] = defaultdict(set)
        sentence_to_entity: Dict[str, Set[str]] = defaultdict(set)
        
        for passage_hash_id, entities in passage_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        
        for sentence, entities in sentence_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        
        return (entity_nodes, sentence_nodes, passage_hash_id_to_entities,
                entity_to_sentence, sentence_to_entity)
    
    def _build_entity_sentence_mappings(self) -> None:
        """Build hash ID mappings between entities and sentences."""
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentences in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id.get(entity)
            if entity_hash_id:
                self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [
                    self.sentence_embedding_store.text_to_hash_id[s]
                    for s in sentences
                    if s in self.sentence_embedding_store.text_to_hash_id
                ]
        
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id.get(sentence)
            if sentence_hash_id:
                self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [
                    self.entity_embedding_store.text_to_hash_id[e]
                    for e in entities
                    if e in self.entity_embedding_store.text_to_hash_id
                ]
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    
    def retrieve(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Retrieve relevant passages for questions.
        
        Args:
            questions: List of question dicts with 'question' and 'answer' keys
            
        Returns:
            List of result dicts with:
                - 'question': The original question
                - 'sorted_passage': List of retrieved passage texts
                - 'sorted_passage_scores': List of retrieval scores
                - 'sorted_passage_metadata': List of metadata dicts with file_source and page_num
                - 'gold_answer': The expected answer (if provided)
        """
        # Cache embeddings for fast retrieval
        self._cache_embeddings()
        
        retrieval_results = []
        
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            
            # Encode question
            question_embedding = self.config.embedding_model.encode(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size
            )
            
            # Get seed entities from question
            seed_data = self._get_seed_entities(question)
            seed_indices, seed_texts, seed_hash_ids, seed_scores = seed_data
            
            if seed_texts:
                # Graph-based retrieval with entities
                sorted_hash_ids, sorted_scores = self._graph_search_with_seed_entities(
                    question_embedding, seed_indices, seed_texts, 
                    seed_hash_ids, seed_scores
                )
                final_hash_ids = sorted_hash_ids[:self.config.retrieval_top_k]
                final_scores = sorted_scores[:self.config.retrieval_top_k]
                final_passages = [
                    strip_passage_prefix(
                        self.passage_embedding_store.hash_id_to_text[hid]
                    )
                    for hid in final_hash_ids
                ]
                final_metadata = [
                    self.passage_embedding_store.get_metadata(hid)
                    for hid in final_hash_ids
                ]
            else:
                # Fallback to dense retrieval
                sorted_indices, sorted_scores = self._dense_passage_retrieval(
                    question_embedding
                )
                final_indices = sorted_indices[:self.config.retrieval_top_k]
                final_scores = sorted_scores[:self.config.retrieval_top_k]
                final_passages = [
                    strip_passage_prefix(self.passage_embedding_store.texts[idx])
                    for idx in final_indices
                ]
                # Get hash IDs from indices for metadata lookup
                final_hash_ids = [
                    self.passage_embedding_store.hash_ids[idx]
                    for idx in final_indices
                ]
                final_metadata = [
                    self.passage_embedding_store.get_metadata(hid)
                    for hid in final_hash_ids
                ]
            
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_scores,
                "sorted_passage_metadata": final_metadata,
                "gold_answer": question_info.get("answer", "")
            }
            retrieval_results.append(result)
        
        return retrieval_results
    
    def _cache_embeddings(self) -> None:
        """Cache embeddings in memory for fast retrieval."""
        self._entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self._entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        
        self._passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self._passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        
        self._sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self._sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
    
    def _get_seed_entities(
        self, 
        question: str
    ) -> Tuple[List[int], List[str], List[str], List[float]]:
        """Extract entities from question and find matching indexed entities."""
        question_entities = list(self.spacy_ner.question_ner(question))
        
        if not question_entities:
            return [], [], [], []
        
        # Encode question entities
        question_entity_embeddings = self.config.embedding_model.encode(
            question_entities,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size
        )
        
        # Find best matching indexed entities
        if len(self._entity_embeddings) == 0:
            return [], [], [], []
            
        similarities = np.dot(self._entity_embeddings, question_entity_embeddings.T)
        
        seed_indices = []
        seed_texts = []
        seed_hash_ids = []
        seed_scores = []
        
        for query_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_idx]
            best_idx = np.argmax(entity_scores)
            best_score = entity_scores[best_idx]
            best_hash_id = self._entity_hash_ids[best_idx]
            best_text = self.entity_embedding_store.hash_id_to_text[best_hash_id]
            
            seed_indices.append(best_idx)
            seed_texts.append(best_text)
            seed_hash_ids.append(best_hash_id)
            seed_scores.append(float(best_score))
        
        return seed_indices, seed_texts, seed_hash_ids, seed_scores
    
    def _graph_search_with_seed_entities(
        self,
        question_embedding: np.ndarray,
        seed_indices: List[int],
        seed_entities: List[str],
        seed_hash_ids: List[str],
        seed_scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        """Perform graph-based search starting from seed entities."""
        # Calculate entity weights via iterative expansion
        entity_weights, activated_entities = self._calculate_entity_scores(
            question_embedding, seed_indices, seed_entities, 
            seed_hash_ids, seed_scores
        )
        
        # Calculate passage weights
        passage_weights = self._calculate_passage_scores(
            question_embedding, activated_entities
        )
        
        # Combined weights
        node_weights = entity_weights + passage_weights
        
        # Run Personalized PageRank
        return self.graph_builder.run_personalized_pagerank(
            node_weights, 
            damping=self.config.damping
        )
    
    def _calculate_entity_scores(
        self,
        question_embedding: np.ndarray,
        seed_indices: List[int],
        seed_entities: List[str],
        seed_hash_ids: List[str],
        seed_scores: List[float]
    ) -> Tuple[np.ndarray, Dict[str, Tuple[int, float, int]]]:
        """Calculate entity weights via iterative graph expansion."""
        activated_entities: Dict[str, Tuple[int, float, int]] = {}
        num_nodes = len(self.graph_builder.node_name_to_vertex_idx)
        entity_weights = np.zeros(num_nodes)
        
        # Initialize with seed entities
        for idx, entity, hash_id, score in zip(
            seed_indices, seed_entities, seed_hash_ids, seed_scores
        ):
            activated_entities[hash_id] = (idx, score, 1)
            if hash_id in self.graph_builder.node_name_to_vertex_idx:
                node_idx = self.graph_builder.node_name_to_vertex_idx[hash_id]
                entity_weights[node_idx] = score
        
        # Iterative expansion
        used_sentence_hash_ids: Set[str] = set()
        current_entities = activated_entities.copy()
        iteration = 1
        
        while current_entities and iteration < self.config.max_iterations:
            new_entities: Dict[str, Tuple[int, float, int]] = {}
            
            for entity_hash_id, (entity_idx, entity_score, tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                
                # Get sentences containing this entity
                sentence_hash_ids = [
                    sid for sid in self.entity_hash_id_to_sentence_hash_ids.get(entity_hash_id, [])
                    if sid not in used_sentence_hash_ids
                ]
                
                if not sentence_hash_ids:
                    continue
                
                # Get sentence embeddings
                sentence_indices = [
                    self.sentence_embedding_store.hash_id_to_idx[sid]
                    for sid in sentence_hash_ids
                    if sid in self.sentence_embedding_store.hash_id_to_idx
                ]
                
                if not sentence_indices:
                    continue
                    
                sentence_embeddings = self._sentence_embeddings[sentence_indices]
                
                # Calculate similarities
                q_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                similarities = np.dot(sentence_embeddings, q_emb).flatten()
                
                # Get top sentences
                top_indices = np.argsort(similarities)[::-1][:self.config.top_k_sentence]
                
                for top_idx in top_indices:
                    if top_idx >= len(sentence_hash_ids):
                        continue
                        
                    top_sentence_hash_id = sentence_hash_ids[top_idx]
                    top_sentence_score = similarities[top_idx]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    
                    # Find entities in this sentence
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids.get(
                        top_sentence_hash_id, []
                    )
                    
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_score = entity_score * top_sentence_score
                        
                        if next_score < self.config.iteration_threshold:
                            continue
                        
                        if next_entity_hash_id in self.graph_builder.node_name_to_vertex_idx:
                            next_node_idx = self.graph_builder.node_name_to_vertex_idx[next_entity_hash_id]
                            entity_weights[next_node_idx] += next_score
                            new_entities[next_entity_hash_id] = (next_node_idx, next_score, iteration + 1)
            
            activated_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        
        return entity_weights, activated_entities
    
    def _calculate_passage_scores(
        self,
        question_embedding: np.ndarray,
        activated_entities: Dict[str, Tuple[int, float, int]]
    ) -> np.ndarray:
        """Calculate passage weights based on dense retrieval and entity bonuses."""
        num_nodes = len(self.graph_builder.node_name_to_vertex_idx)
        passage_weights = np.zeros(num_nodes)
        
        # Get dense retrieval scores
        dpr_indices, dpr_scores = self._dense_passage_retrieval(question_embedding)
        dpr_scores_normalized = min_max_normalize(np.array(dpr_scores))
        
        for i, passage_idx in enumerate(dpr_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[passage_idx]
            dpr_score = dpr_scores_normalized[i]
            
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            
            # Calculate entity bonus
            for entity_hash_id, (_, entity_score, tier) in activated_entities.items():
                entity_text = self.entity_embedding_store.hash_id_to_text.get(entity_hash_id, "")
                entity_lower = entity_text.lower()
                occurrences = passage_text_lower.count(entity_lower)
                
                if occurrences > 0:
                    denom = max(tier, 1)
                    bonus = entity_score * math.log(1 + occurrences) / denom
                    total_entity_bonus += bonus
            
            # Combined score
            passage_score = (
                self.config.passage_ratio * dpr_score + 
                math.log(1 + total_entity_bonus)
            )
            
            if passage_hash_id in self.graph_builder.node_name_to_vertex_idx:
                node_idx = self.graph_builder.node_name_to_vertex_idx[passage_hash_id]
                passage_weights[node_idx] = passage_score * self.config.passage_node_weight
        
        return passage_weights
    
    def _dense_passage_retrieval(
        self, 
        question_embedding: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """Perform dense passage retrieval via embedding similarity."""
        q_emb = question_embedding.reshape(1, -1)
        similarities = np.dot(self._passage_embeddings, q_emb.T).flatten()
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_scores = similarities[sorted_indices].tolist()
        return sorted_indices, sorted_scores
    
    # =========================================================================
    # QUESTION ANSWERING
    # =========================================================================
    
    def qa(
        self, 
        questions: List[Dict[str, str]],
        system_prompt: str = None
    ) -> List[Dict[str, Any]]:
        """Perform end-to-end question answering.
        
        This method:
        1. Retrieves relevant passages
        2. Generates answers using the configured LLM
        
        Args:
            questions: List of question dicts with 'question' and 'answer' keys
            system_prompt: Optional custom system prompt for LLM
            
        Returns:
            List of result dicts with retrieval results plus 'pred_answer'
        """
        if self.llm_model is None:
            raise ValueError(
                "LLM model not configured. Set llm_model in LinearRAGConfig."
            )
        
        qa_start = time.time()
        num_questions = len(questions)
        
        # Retrieval
        logger.info(f"Starting retrieval for {num_questions} questions...")
        retrieval_start = time.time()
        retrieval_results = self.retrieve(questions)
        retrieval_time = time.time() - retrieval_start
        
        logger.info(
            f"[Timing] Retrieval: {retrieval_time:.2f}s total, "
            f"{retrieval_time/num_questions:.2f}s per question"
        )
        
        # Build prompts
        prompt_system = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        all_messages = []
        
        for result in retrieval_results:
            prompt_user = ""
            for passage in result["sorted_passage"]:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {result['question']}\n Thought: "
            
            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)
        
        # Generate answers
        logger.info(f"Starting generation for {num_questions} questions...")
        generation_start = time.time()
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_responses = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Generation"
            ))
        
        generation_time = time.time() - generation_start
        
        logger.info(
            f"[Timing] Generation: {generation_time:.2f}s total, "
            f"{generation_time/num_questions:.2f}s per question"
        )
        
        # Extract answers
        for response, result in zip(all_responses, retrieval_results):
            try:
                pred_answer = response.split('Answer:')[1].strip()
            except (IndexError, AttributeError):
                pred_answer = response
            result["pred_answer"] = pred_answer
        
        total_time = time.time() - qa_start
        logger.info(
            f"[Timing] Total QA: {total_time:.2f}s total, "
            f"{total_time/num_questions:.2f}s per question"
        )
        
        return retrieval_results

