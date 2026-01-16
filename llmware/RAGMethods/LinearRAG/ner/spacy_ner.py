"""SpaCy-based Named Entity Recognition for LinearRAG.

Extracts entities from passages and questions for graph construction.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import logging

import spacy

logger = logging.getLogger(__name__)


class SpacyNER:
    """Named Entity Recognition using SpaCy models.
    
    Extracts named entities from passages and questions, filtering out
    ordinal and cardinal numbers which are typically not useful for
    graph-based retrieval.
    
    Args:
        spacy_model: Name of SpaCy model to load (e.g., 'en_core_web_trf')
    """
    
    # Entity types to exclude from extraction
    EXCLUDED_LABELS = {"ORDINAL", "CARDINAL"}
    
    def __init__(self, spacy_model: str = "en_core_web_trf"):
        self.spacy_model = spacy.load(spacy_model)
        logger.info(f"Loaded SpaCy model: {spacy_model}")
    
    def batch_ner(
        self,
        hash_id_to_passage: Dict[str, str],
        max_workers: int = 4
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Perform batch NER on multiple passages.
        
        Args:
            hash_id_to_passage: Mapping of passage hash IDs to passage text
            max_workers: Number of workers for batch processing
            
        Returns:
            Tuple of:
                - passage_hash_id_to_entities: Map of passage hash ID to list of entities
                - sentence_to_entities: Map of sentence text to list of entities
        """
        passage_list = list(hash_id_to_passage.values())
        hash_ids = list(hash_id_to_passage.keys())
        
        # Calculate batch size based on workers
        batch_size = max(1, len(passage_list) // max_workers)
        
        # Process all passages with SpaCy's efficient pipe
        docs_list = self.spacy_model.pipe(passage_list, batch_size=batch_size)
        
        passage_hash_id_to_entities: Dict[str, List[str]] = {}
        sentence_to_entities: Dict[str, List[str]] = defaultdict(list)
        
        for idx, doc in enumerate(docs_list):
            passage_hash_id = hash_ids[idx]
            
            single_passage_entities, single_sentence_entities = \
                self.extract_entities_sentences(doc, passage_hash_id)
            
            passage_hash_id_to_entities.update(single_passage_entities)
            
            # Merge sentence entities, avoiding duplicates
            for sent, ents in single_sentence_entities.items():
                for e in ents:
                    if e not in sentence_to_entities[sent]:
                        sentence_to_entities[sent].append(e)
        
        return passage_hash_id_to_entities, dict(sentence_to_entities)
    
    def extract_entities_sentences(
        self,
        doc,
        passage_hash_id: str
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Extract entities and their sentence contexts from a SpaCy doc.
        
        Args:
            doc: SpaCy Doc object
            passage_hash_id: Hash ID of the passage
            
        Returns:
            Tuple of:
                - passage_hash_id_to_entities: Single-entry dict mapping passage to entities
                - sentence_to_entities: Map of sentence text to entities in that sentence
        """
        sentence_to_entities: Dict[str, List[str]] = defaultdict(list)
        unique_entities: Set[str] = set()
        
        for ent in doc.ents:
            # Skip ordinal and cardinal numbers
            if ent.label_ in self.EXCLUDED_LABELS:
                continue
            
            sent_text = ent.sent.text
            ent_text = ent.text
            
            # Add to sentence mapping if not already present
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            
            unique_entities.add(ent_text)
        
        passage_hash_id_to_entities = {passage_hash_id: list(unique_entities)}
        
        return passage_hash_id_to_entities, dict(sentence_to_entities)
    
    def question_ner(self, question: str) -> Set[str]:
        """Extract entities from a question.
        
        Args:
            question: Question text
            
        Returns:
            Set of lowercased entity strings found in the question
        """
        doc = self.spacy_model(question)
        question_entities: Set[str] = set()
        
        for ent in doc.ents:
            # Skip ordinal and cardinal numbers
            if ent.label_ in self.EXCLUDED_LABELS:
                continue
            question_entities.add(ent.text.lower())
        
        return question_entities

