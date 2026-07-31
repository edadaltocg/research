# EBM Sudoku solver

## Sudoku

TODO: paraphrase this.

The game in its current form was invented by American Howard Garns in 1979 and published by Dell Magazines as "Numbers in Place." In 1984, Maki Kaji of Japan published it in the magazine of his puzzle company Nikoli. He gave the game its modern name of Sudoku, which means "Single Numbers."

The standard version of Sudoku consists of a 9×9 square grid containing 81 cells. The grid is subdivided into nine 3×3 blocks. Some of the 81 cells are filled in with numbers from the set {1,2,3,4,5,6,7,8,9}. These filled-in cells are called givens. The goal is to fill in the whole grid using the nine digits so that each row, each column, and each block contains each number exactly once.

The above-described puzzle is called a Sudoku of rank 3. A Sudoku of rank n is an n2×n2 square grid, subdivided into n2 blocks, each of size n×n. The numbers used to fill the grid in are 1, 2, 3, ..., n2.

```
+-------+-------+-------+
|C1C2C3 |C4C5C6 |C7C8C9 |
| . . . | . . . | . . . |
| . . . | . . . | . . . |
+-------+-------+-------+
|C1C2C3 |C4C5C6 |C7C8C9 |
| . . . | . . . | . . . |
| . . . | . . . | . . . |
+-------+-------+-------+
|C1C2C3 |C4C5C6 |C7C8C9 |
| . . . | . . . | . . . |
| . . . | . . . | . . . |
+-------+-------+-------+

+-------+-------+-------+
|R1R1R1 |R1R1R1 |R1R1R1 |
|R2R2R2 |R2R2R2 |R2R2R2 |
|R3R3R3 |R3R3R3 |R3R3R3 |
+-------+-------+-------+
|R4R4R4 |R4R4R4 |R4R4R4 |
|R5R5R5 |R5R5R5 |R5R5R5 |
|R6R6R6 |R6R6R6 |R6R6R6 |
+-------+-------+-------+
|R7R7R7 |R7R7R7 |R7R7R7 |
|R8R8R8 |R8R8R8 |R8R8R8 |
|R9R9R9 |R9R9R9 |R9R9R9 |
+-------+-------+-------+

+-------+-------+-------+
| . . . | . . . | . . . |
| . B1. | . B2. | . B3. |
| . . . | . . . | . . . |
+-------+-------+-------+
| . . . | . . . | . . . |
| . B4. | . B5. | . B6. |
| . . . | . . . | . . . |
+-------+-------+-------+
| . . . | . . . | . . . |
| . B7. | . B8. | . B9. |
| . . . | . . . | . . . |
+-------+-------+-------+

+-------+-------+-------+
| . . . | 8 . . | . . . |
| 4 . . | . 1 5 | . 3 . |
| . 2 9 | . 4 . | 5 1 8 |
+-------+-------+-------+
| . 4 . | . . . | 1 2 . |
| . . . | 6 . 2 | . . . |
| . 3 2 | . . . | . 9 . |
+-------+-------+-------+
| 6 9 3 | . 5 . | 8 7 . |
| . 5 . | 4 8 . | . . 1 |
| . . . | . . 3 | . . . |
+-------+-------+-------+

+-------+-------+-------+
| 3 1 5 | 8 2 7 | 9 4 6 |
| 4 6 8 | 9 1 5 | 7 3 2 |
| 7 2 9 | 3 4 6 | 5 1 8 |
+-------+-------+-------+
| 9 4 6 | 5 3 8 | 1 2 7 |
| 5 7 1 | 6 9 2 | 4 8 3 |
| 8 3 2 | 1 7 4 | 6 9 5 |
+-------+-------+-------+
| 6 9 3 | 2 5 1 | 8 7 4 |
| 2 5 7 | 4 8 9 | 3 6 1 |
| 1 8 4 | 7 6 3 | 2 5 9 |
+-------+-------+-------+
```

As the rank of a Sudoku increases from n to n+1, the extra computational time needed to find a solution increases quite fast. This places the game of solving rank-n Sudoku puzzles in a class of problems that computer scientists have named NP-complete.

An NP-complete problem satisfies the following two properties:

1. Any solution to the problem can be checked relatively quickly, i.e. in polynomial time.
2. If the problem can be solved relatively quickly, then so can every problem that satisfies property (1).

Even though checking a solution of an NP-complete problem can be done relatively quickly and easily, there is no known algorithm for finding a solution to begin with, because the time needed for computation increases so fast compared to the size of the problem.

## Dimensionality of the problem

Complete valid Sudoku grids: ~6.67x10^21 (Felgenhauer & Jarvis, 2005)

## EBM

EBM stands for energy-based-model.

```
```

## References

- [The Math Behind Sudoku](https://pi.math.cornell.edu/~mec/Summer2009/Mahmood/Home.html)
