"""Graph construction for LinearRAG using igraph.

Builds an entity-passage graph with weighted edges for Personalized PageRank.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import re
import logging

import numpy as np
import igraph as ig

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds and manages the entity-passage graph for LinearRAG.
    
    The graph contains:
    - Passage nodes (document chunks)
    - Entity nodes (named entities extracted via NER)
    - Edges between passages and their entities (weighted by frequency)
    - Edges between adjacent passages (sequential order)
    
    Args:
        passage_embedding_store: Store for passage embeddings
        entity_embedding_store: Store for entity embeddings
    """
    
    def __init__(
        self,
        passage_embedding_store: Any,
        entity_embedding_store: Any
    ):
        self.passage_embedding_store = passage_embedding_store
        self.entity_embedding_store = entity_embedding_store
        
        # Initialize empty graph
        self.graph = ig.Graph(directed=False)
        
        # Edge weight tracking
        self.node_to_node_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Index mappings (populated after build)
        self.node_name_to_vertex_idx: Dict[str, int] = {}
        self.vertex_idx_to_node_name: Dict[int, str] = {}
        self.passage_node_indices: List[int] = []
    
    def build_graph(
        self,
        passage_hash_id_to_entities: Dict[str, Set[str]]
    ) -> None:
        """Build the complete graph from passage-entity mappings.
        
        Args:
            passage_hash_id_to_entities: Map of passage hash IDs to entity sets
        """
        # Add entity-passage edges
        self._add_entity_to_passage_edges(passage_hash_id_to_entities)
        
        # Add adjacent passage edges
        self._add_adjacent_passage_edges()
        
        # Build the igraph structure
        self._augment_graph()
        
        logger.info(
            f"Built graph with {self.graph.vcount()} nodes and "
            f"{self.graph.ecount()} edges"
        )
    
    def _add_entity_to_passage_edges(
        self,
        passage_hash_id_to_entities: Dict[str, Set[str]]
    ) -> None:
        """Add weighted edges between passages and their entities.
        
        Edge weight is based on entity frequency in the passage.
        
        Args:
            passage_hash_id_to_entities: Map of passage hash IDs to entity sets
        """
        passage_to_entity_count: Dict[Tuple[str, str], int] = {}
        passage_to_all_score: Dict[str, int] = defaultdict(int)
        
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text.get(passage_hash_id, "")
            
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id.get(entity)
                if entity_hash_id is None:
                    continue
                    
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        
        # Normalize by total entity count in passage
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            total = passage_to_all_score[passage_hash_id]
            if total > 0:
                score = count / total
                self.node_to_node_stats[passage_hash_id][entity_hash_id] = score
    
    def _add_adjacent_passage_edges(self) -> None:
        """Add edges between sequentially adjacent passages.
        
        Assumes passages are prefixed with index like '0:', '1:', etc.
        """
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        
        # Extract passages with numeric indices
        indexed_items = []
        for node_key, text in passage_id_to_text.items():
            match = index_pattern.match(text.strip())
            if match:
                indexed_items.append((int(match.group(1)), node_key))
        
        # Sort by index and create edges between adjacent passages
        indexed_items.sort(key=lambda x: x[0])
        
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0
    
    def _augment_graph(self) -> None:
        """Build the igraph structure from collected edges."""
        self._add_nodes()
        self._add_edges()
    
    def _add_nodes(self) -> None:
        """Add all passage and entity nodes to the graph."""
        import igraph as ig
        
        existing_nodes = {
            v["name"]: v 
            for v in self.graph.vs 
            if "name" in v.attributes()
        }
        
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        
        # Add new nodes
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)
        
        # Build index mappings
        self.node_name_to_vertex_idx = {
            v["name"]: v.index 
            for v in self.graph.vs 
            if "name" in v.attributes()
        }
        
        self.vertex_idx_to_node_name = {
            v.index: v["name"]
            for v in self.graph.vs
            if "name" in v.attributes()
        }
        
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id]
            for passage_id in passage_hash_ids
            if passage_id in self.node_name_to_vertex_idx
        ]
    
    def _add_edges(self) -> None:
        """Add all edges with weights to the graph."""
        edges = []
        weights = []
        
        for node_hash_id, neighbors in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in neighbors.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        
        if edges:
            self.graph.add_edges(edges)
            self.graph.es['weight'] = weights
    
    def run_personalized_pagerank(
        self,
        node_weights: np.ndarray,
        damping: float = 0.5
    ) -> Tuple[List[str], List[float]]:
        """Run Personalized PageRank on the graph.
        
        Args:
            node_weights: Reset probability weights for each node
            damping: Damping factor (1 - teleport probability)
            
        Returns:
            Tuple of (sorted_passage_hash_ids, sorted_scores)
        """
        # Clean up weights
        reset_prob = np.where(
            np.isnan(node_weights) | (node_weights < 0),
            0,
            node_weights
        )
        
        # Run PPR
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        # Extract passage scores
        doc_scores = np.array([
            pagerank_scores[idx] 
            for idx in self.passage_node_indices
        ])
        
        # Sort by score
        sorted_indices = np.argsort(doc_scores)[::-1]
        sorted_scores = doc_scores[sorted_indices]
        
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]]
            for i in sorted_indices
        ]
        
        return sorted_passage_hash_ids, sorted_scores.tolist()
    
    def save_graph(self, filepath: str) -> None:
        """Save graph to GraphML file.
        
        Args:
            filepath: Path to save GraphML file
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.graph.write_graphml(filepath)
        logger.info(f"Saved graph to {filepath}")
    
    def load_graph(self, filepath: str) -> None:
        """Load graph from GraphML file.
        
        Args:
            filepath: Path to GraphML file
        """
        import igraph as ig
        self.graph = ig.Graph.Read_GraphML(filepath)
        
        # Rebuild index mappings
        self.node_name_to_vertex_idx = {
            v["name"]: v.index
            for v in self.graph.vs
            if "name" in v.attributes()
        }
        
        self.vertex_idx_to_node_name = {
            v.index: v["name"]
            for v in self.graph.vs
            if "name" in v.attributes()
        }
        
        logger.info(f"Loaded graph from {filepath}")

