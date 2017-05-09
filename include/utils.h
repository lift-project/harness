#ifndef EXECUTOR_UTILS_H
#define EXECUTOR_UTILS_H

template <typename T> using Matrix = std::vector<T>;

template <typename T>
void transpose(std::vector<T> &matrix, const size_t rows, const size_t cols) {
  std::vector<T> matrixT(rows * cols);

  for (unsigned y = 0; y < rows; ++y)
    for (unsigned x = 0; x < cols; ++x)
      matrixT[x * rows + y] = matrix[y * cols + x];

  std::swap(matrixT, matrix);
}

#endif // EXECUTOR_UTILS_H
