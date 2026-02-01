"""
Chess-aware position embeddings.

Raw bitboard (768-dim) finds near-identical positions only.
This module adds strategic features so similar *types* of positions
cluster together: same pawn structure, similar king safety, etc.

Total vector: 768 (bitboard) + 64 (features) = 832 dims.
"""

import chess
import numpy as np

VECTOR_DIM = 832


def encode_board(board):
    """Encode board as 832-dim vector: bitboard + strategic features."""
    vec = np.zeros(VECTOR_DIM, dtype=np.float32)

    # --- 768 dims: raw bitboard (piece x square) ---
    piece_offset = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 64,
        (chess.BISHOP, chess.WHITE): 128, (chess.ROOK, chess.WHITE): 192,
        (chess.QUEEN, chess.WHITE): 256, (chess.KING, chess.WHITE): 320,
        (chess.PAWN, chess.BLACK): 384, (chess.KNIGHT, chess.BLACK): 448,
        (chess.BISHOP, chess.BLACK): 512, (chess.ROOK, chess.BLACK): 576,
        (chess.QUEEN, chess.BLACK): 640, (chess.KING, chess.BLACK): 704,
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            vec[piece_offset[(piece.piece_type, piece.color)] + sq] = 1.0

    # --- 64 dims: strategic features (offset 768) ---
    o = 768

    # material balance per piece type (6 dims, normalized)
    for i, pt in enumerate(chess.PIECE_TYPES):
        w = len(board.pieces(pt, chess.WHITE))
        b = len(board.pieces(pt, chess.BLACK))
        vec[o + i] = (w - b) / max(w + b, 1)

    # pawn structure: which files have pawns (16 dims: 8 white + 8 black)
    for color_i, color in enumerate([chess.WHITE, chess.BLACK]):
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            file = chess.square_file(sq)
            vec[o + 6 + color_i * 8 + file] = 1.0

    # king position normalized (4 dims: file+rank per side)
    for color_i, color in enumerate([chess.WHITE, chess.BLACK]):
        ksq = board.king(color)
        if ksq is not None:
            vec[o + 22 + color_i * 2] = chess.square_file(ksq) / 7.0
            vec[o + 23 + color_i * 2] = chess.square_rank(ksq) / 7.0

    # center control: pieces/pawns attacking e4,d4,e5,d5 (8 dims)
    center = [chess.E4, chess.D4, chess.E5, chess.D5]
    for i, sq in enumerate(center):
        w_attacks = bool(board.attackers(chess.WHITE, sq))
        b_attacks = bool(board.attackers(chess.BLACK, sq))
        vec[o + 26 + i * 2] = float(w_attacks)
        vec[o + 27 + i * 2] = float(b_attacks)

    # castling rights (4 dims)
    vec[o + 34] = float(board.has_kingside_castling_rights(chess.WHITE))
    vec[o + 35] = float(board.has_queenside_castling_rights(chess.WHITE))
    vec[o + 36] = float(board.has_kingside_castling_rights(chess.BLACK))
    vec[o + 37] = float(board.has_queenside_castling_rights(chess.BLACK))

    # side to move
    vec[o + 38] = float(board.turn)

    # mobility: legal move count normalized (1 dim)
    vec[o + 39] = len(list(board.legal_moves)) / 60.0

    # piece count total (proxy for game phase: opening/middle/endgame)
    total_pieces = len(board.piece_map())
    vec[o + 40] = total_pieces / 32.0

    # open files: files with no pawns (8 dims)
    for file in range(8):
        has_pawn = False
        for rank in range(8):
            sq = chess.square(file, rank)
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN:
                has_pawn = True
                break
        vec[o + 41 + file] = float(not has_pawn)

    # passed pawns count per side (2 dims)
    for color_i, color in enumerate([chess.WHITE, chess.BLACK]):
        opp = not color
        passed = 0
        for sq in board.pieces(chess.PAWN, color):
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_passed = True
            for adj_file in [file - 1, file, file + 1]:
                if 0 <= adj_file <= 7:
                    for r in range(rank + (1 if color == chess.WHITE else -7),
                                   8 if color == chess.WHITE else rank):
                        s = chess.square(adj_file, r)
                        p = board.piece_at(s)
                        if p and p.piece_type == chess.PAWN and p.color == opp:
                            is_passed = False
                            break
                    if not is_passed:
                        break
            if is_passed:
                passed += 1
        vec[o + 49 + color_i] = passed / 8.0

    # bishop pair (2 dims)
    for color_i, color in enumerate([chess.WHITE, chess.BLACK]):
        vec[o + 51 + color_i] = float(len(board.pieces(chess.BISHOP, color)) >= 2)

    # king pawn shield: pawns on ranks 2-3 near king (2 dims)
    for color_i, color in enumerate([chess.WHITE, chess.BLACK]):
        ksq = board.king(color)
        if ksq is None:
            continue
        kfile = chess.square_file(ksq)
        shield = 0
        shield_ranks = [1, 2] if color == chess.WHITE else [5, 6]
        for f in [kfile - 1, kfile, kfile + 1]:
            if 0 <= f <= 7:
                for r in shield_ranks:
                    s = chess.square(f, r)
                    p = board.piece_at(s)
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        shield += 1
        vec[o + 53 + color_i] = shield / 6.0

    # remaining dims (55-63) reserved, stay 0

    return vec.tolist()
