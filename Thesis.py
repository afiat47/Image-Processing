import math

def evaluate(board):
    """
    Evaluate the current state of the tic-tac-toe board.
    Returns:
        1 if player X wins
        -1 if player O wins
        0 if it's a draw or the game is ongoing
    """
    # Check rows, columns, and diagonals
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != ' ':
            return 1 if board[row][0] == 'X' else -1
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return 1 if board[0][col] == 'X' else -1
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return 1 if board[0][0] == 'X' else -1
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return 1 if board[0][2] == 'X' else -1
    # Check for draw
    if ' ' not in board[0] + board[1] + board[2]:
        return 0
    return None

def minimax(board, depth, is_maximizing):
    """
    Minimax algorithm implementation.
    Args:
        board: Current state of the tic-tac-toe board
        depth: Current depth of the search tree
        is_maximizing: True if it's maximizing player's turn, False otherwise
    Returns:
        Best score for the current state of the board
    """
    result = evaluate(board)
    if result is not None:
        return result

    if is_maximizing:
        best_score = -math.inf
        for row in range(3):
            for col in range(3):
                if board[row][col] == ' ':
                    board[row][col] = 'X'
                    score = minimax(board, depth + 1, False)
                    board[row][col] = ' '
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for row in range(3):
            for col in range(3):
                if board[row][col] == ' ':
                    board[row][col] = 'O'
                    score = minimax(board, depth + 1, True)
                    board[row][col] = ' '
                    best_score = min(score, best_score)
        return best_score

def find_best_move(board):
    """
    Find the best move for the maximizing player (X).
    Args:
        board: Current state of the tic-tac-toe board
    Returns:
        Best move represented as a tuple (row, col)
    """
    best_score = -math.inf
    best_move = None
    for row in range(3):
        for col in range(3):
            if board[row][col] == ' ':
                board[row][col] = 'X'
                score = minimax(board, 0, False)
                board[row][col] = ' '
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
    return best_move

# Example usage:
if __name__ == "__main__":
    board = [[' ', ' ', ' '],
             [' ', ' ', ' '],
             [' ', ' ', ' ']]

    print("Initial Board:")
    for row in board:
        print(row)

    while True:
        player_move = input("Enter your move (row, col): ").split(',')
        row, col = int(player_move[0]), int(player_move[1])
        if board[row][col] != ' ':
            print("Invalid move! Cell already taken.")
            continue
        board[row][col] = 'O'
        print("After your move:")
        for row in board:
            print(row)

        if evaluate(board) is not None:
            break

        ai_move = find_best_move(board)
        row, col = ai_move[0], ai_move[1]
        board[row][col] = 'X'
        print("After AI's move:")
        for row in board:
            print(row)

        if evaluate(board) is not None:
            break
