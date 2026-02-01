"""
Retrieval-based chess engine. ~100 lines. Looks up positions in Qdrant Cloud,
falls back to minimal search when no good match exists.
"""

import os
import random
import chess
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from embeddings import encode_board, VECTOR_DIM

load_dotenv()

COLLECTION = "magnus_chess"


class RetrievalEngine:
    def __init__(self):
        self.client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
        self.hits = 0
        self.misses = 0

    def lookup(self, board, threshold=0.80):
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=encode_board(board),
            limit=1,
        ).points
        if not results:
            return None
        top = results[0]
        if top.score >= threshold:
            return top.payload["best_move"], top.payload["score"], top.score
        return None

    def get_move(self, board):
        result = self.lookup(board)
        if result:
            move_uci, score, sim = result
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                self.hits += 1
                return move

        self.misses += 1
        return random.choice(list(board.legal_moves))

    def stats(self):
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0
        return f"Retrieval: {self.hits}/{total} hits ({rate:.0f}%)"
