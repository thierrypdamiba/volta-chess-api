"""
Game loop: Sunfish (white) vs Retrieval Engine (black).
Saves each game as a PGN file in the games/ directory.
"""

import chess
import chess.pgn
import sys
import os
from datetime import datetime

from sunfish import get_move as sunfish_move

GAMES_DIR = os.path.join(os.path.dirname(__file__), "games")


def play_game(use_retrieval=False):
    board = chess.Board()
    retrieval = None
    black_name = "Sunfish"

    if use_retrieval:
        from retrieval_engine import RetrievalEngine
        retrieval = RetrievalEngine()
        black_name = "RetrievalEngine"

    game = chess.pgn.Game()
    game.headers["White"] = "Sunfish"
    game.headers["Black"] = black_name
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    node = game

    move_num = 0
    print(f"\n{'='*40}")
    print(f"Sunfish (White) vs {black_name} (Black)")
    print(f"{'='*40}\n")
    print(board)
    print()

    while not board.is_game_over():
        move_num += 1
        if board.turn == chess.WHITE:
            print(f"Move {move_num} (White/Sunfish) thinking...")
            move = sunfish_move(board, depth=4)
        else:
            if retrieval:
                print(f"Move {move_num} (Black/Retrieval) thinking...")
                move = retrieval.get_move(board)
            else:
                print(f"Move {move_num} (Black/Sunfish) thinking...")
                move = sunfish_move(board, depth=4)

        if move is None:
            print("No move found, game over.")
            break

        san = board.san(move)
        node = node.add_variation(move)
        board.push(move)
        side = "White" if not board.turn else "Black"
        print(f"  {side}: {san}")
        print(board)
        print()

    result = board.result()
    reason = _game_over_reason(board)
    game.headers["Result"] = result
    game.headers["Termination"] = reason

    print(f"Result: {result}")
    print(f"Reason: {reason}")
    if retrieval:
        print(retrieval.stats())

    _save_pgn(game)
    return result


def _save_pgn(game):
    os.makedirs(GAMES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    white = game.headers["White"]
    black = game.headers["Black"]
    path = os.path.join(GAMES_DIR, f"{ts}_{white}_vs_{black}.pgn")
    with open(path, "w") as f:
        print(game, file=f)
    print(f"Game saved to {path}")


def _game_over_reason(board):
    if board.is_checkmate():
        return "Checkmate"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Insufficient material"
    if board.is_fifty_moves():
        return "Fifty-move rule"
    if board.is_repetition():
        return "Threefold repetition"
    return "Unknown"


if __name__ == "__main__":
    use_retrieval = "--retrieval" in sys.argv
    play_game(use_retrieval=use_retrieval)
