"""
Benchmark: Sunfish vs Retrieval Engine over multiple games.
Tracks wins, retrieval hit rate, and move times.
Saves results to benchmarks/ as JSON.
"""

import json
import os
import time
import chess
import chess.pgn
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

from sunfish import get_move as sunfish_move
from retrieval_engine import RetrievalEngine

BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")


def play_game(retrieval, game_num, verbose=False):
    board = chess.Board()
    move_times_sunfish = []
    move_times_retrieval = []
    moves = 0

    pgn_game = chess.pgn.Game()
    pgn_game.headers["White"] = "Sunfish"
    pgn_game.headers["Black"] = "RetrievalEngine"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    node = pgn_game
    prev_hits = 0

    while not board.is_game_over():
        moves += 1
        if board.turn == chess.WHITE:
            t0 = time.time()
            move = sunfish_move(board, depth=4)
            move_times_sunfish.append(time.time() - t0)
        else:
            t0 = time.time()
            move = retrieval.get_move(board)
            move_times_retrieval.append(time.time() - t0)

        if move is None:
            if verbose:
                print(f"\n  Retrieval ran out of knowledge at move {moves}")
            break

        if verbose:
            san = board.san(move)
            if board.turn == chess.WHITE:
                print(f"  W: {san}", end="")
            else:
                hits_now = retrieval.hits
                tag = "hit" if hits_now > prev_hits else "rand"
                prev_hits = hits_now
                print(f"  B: {san} [{tag}]", end="")
            if moves % 4 == 0:
                print()

        node = node.add_variation(move)
        board.push(move)

    result = board.result()
    if verbose:
        print()
    reason = _reason(board)
    print(f"  Game {game_num}: {result} ({reason}) in {moves} moves")

    pgn_game.headers["Result"] = result
    pgn_game.headers["Termination"] = reason

    hits_before = retrieval.hits
    misses_before = retrieval.misses

    return {
        "result": result,
        "reason": reason,
        "moves": moves,
        "avg_sunfish_ms": _avg_ms(move_times_sunfish),
        "avg_retrieval_ms": _avg_ms(move_times_retrieval),
        "hits": hits_before,
        "misses": misses_before,
        "pgn": str(pgn_game),
    }


def _avg_ms(times):
    return (sum(times) / len(times) * 1000) if times else 0


def _reason(board):
    if board.is_checkmate(): return "checkmate"
    if board.is_stalemate(): return "stalemate"
    if board.is_insufficient_material(): return "insufficient"
    if board.is_fifty_moves(): return "50-move"
    if board.is_repetition(): return "repetition"
    return "unknown"


def _run_single_game(game_num, verbose=False):
    """Run a single game with its own RetrievalEngine instance."""
    retrieval = RetrievalEngine()
    return play_game(retrieval, game_num, verbose=verbose)


def run_benchmark(num_games=10, verbose=False, workers=None):
    if workers is None:
        workers = num_games
    print(f"Benchmark: Sunfish (W) vs retrieval + random fallback (B) x {num_games} games ({workers} parallel)")
    print(f"{'='*50}")

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    total_moves = 0
    total_sunfish_ms = 0
    total_retrieval_ms = 0

    games = [None] * num_games
    total_hits = 0
    total_misses = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_single_game, i + 1, verbose): i
            for i in range(num_games)
        }
        for future in tqdm(as_completed(futures), total=num_games, desc="Games", unit="game"):
            idx = futures[future]
            game = future.result()
            games[idx] = game
            results[game["result"]] += 1
            total_moves += game["moves"]
            total_sunfish_ms += game["avg_sunfish_ms"]
            total_retrieval_ms += game["avg_retrieval_ms"]
            total_hits += game["hits"]
            total_misses += game["misses"]

    total = total_hits + total_misses
    hit_rate = (total_hits / total * 100) if total else 0

    print(f"\n{'='*50}")
    print(f"Results ({num_games} games):")
    print(f"  Sunfish wins:    {results['1-0']}")
    print(f"  Retrieval wins:  {results['0-1']}")
    print(f"  Draws:           {results['1/2-1/2']}")
    print(f"  Avg moves/game:  {total_moves / num_games:.0f}")
    print(f"  Avg move time:")
    print(f"    Sunfish:       {total_sunfish_ms / num_games:.0f} ms")
    print(f"    Retrieval:     {total_retrieval_ms / num_games:.0f} ms")
    print(f"  Retrieval: {total_hits}/{total} hits ({hit_rate:.0f}%)")

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "retrieval + random fallback",
        "num_games": num_games,
        "workers": workers,
        "summary": {
            "sunfish_wins": results["1-0"],
            "retrieval_wins": results["0-1"],
            "draws": results["1/2-1/2"],
            "avg_moves": round(total_moves / num_games),
            "avg_sunfish_ms": round(total_sunfish_ms / num_games),
            "avg_retrieval_ms": round(total_retrieval_ms / num_games),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate_pct": round(hit_rate, 1),
        },
        "games": games,
    }

    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(BENCHMARKS_DIR, f"benchmark_{ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark saved to {path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    verbose = "--verbose" in sys.argv
    run_benchmark(n, verbose)
