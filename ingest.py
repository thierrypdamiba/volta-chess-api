"""
Ingest last 10 Magnus Carlsen (DrNykterstein) games from Lichess into Qdrant Cloud.

Usage: uv run ingest.py
"""

import io
import os
import chess
import chess.pgn
import hashlib
import uuid
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    BinaryQuantization, BinaryQuantizationConfig,
)

load_dotenv()

from embeddings import encode_board, VECTOR_DIM

COLLECTION = "magnus_chess"
LICHESS_USER = "DrNykterstein"
NUM_GAMES = 100


def board_hash(fen):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, fen))


def get_client():
    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]
    return QdrantClient(url=url, api_key=api_key)


def create_collection(client):
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION in collections:
        print(f"Collection '{COLLECTION}' exists, skipping creation.")
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        quantization_config=BinaryQuantization(
            binary=BinaryQuantizationConfig(always_ram=True),
        ),
    )
    print(f"Created collection '{COLLECTION}' with binary quantization.")


def fetch_games():
    """Fetch last 10 Magnus games from Lichess API as PGN."""
    url = f"https://lichess.org/api/games/user/{LICHESS_USER}"
    params = {"max": NUM_GAMES, "pgnInJson": False}
    headers = {"Accept": "application/x-chess-pgn"}
    print(f"Fetching last {NUM_GAMES} games for {LICHESS_USER} from Lichess...")
    resp = httpx.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def result_to_score(result, is_white_turn):
    if result == "1-0":
        return 100 if is_white_turn else -100
    elif result == "0-1":
        return -100 if is_white_turn else 100
    return 0


def ingest(client, pgn_text):
    seen = set()
    points = []
    games = 0

    f = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break

        result = game.headers.get("Result", "*")
        if result == "*":
            continue

        games += 1
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        is_magnus_white = LICHESS_USER.lower() in white.lower()
        is_magnus_black = LICHESS_USER.lower() in black.lower()

        print(f"  Game {games}: {white} vs {black} ({result})")

        board = game.board()
        for move_num, move in enumerate(game.mainline_moves()):
            magnus_turn = (board.turn == chess.WHITE and is_magnus_white) or \
                          (board.turn == chess.BLACK and is_magnus_black)

            if magnus_turn:
                fen = board.fen()
                h = board_hash(fen)
                if h not in seen:
                    seen.add(h)
                    points.append(PointStruct(
                        id=h,
                        vector=encode_board(board),
                        payload={
                            "fen": fen,
                            "best_move": move.uci(),
                            "score": result_to_score(result, board.turn),
                            "move_number": move_num // 2 + 1,
                            "source": f"{white} vs {black}",
                        },
                    ))
            board.push(move)

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    print(f"\nDone. {len(points)} positions from {games} games stored in '{COLLECTION}'.")


if __name__ == "__main__":
    client = get_client()
    create_collection(client)
    pgn_text = fetch_games()
    ingest(client, pgn_text)
