#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "mmio.h"
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <tuple>
#include <vector>

// #include <SkelCL/Vector.h>
// #include "sparseStructs.h"

// An OpenCL sparse matrix is one that we can build kernel args directly from
template <typename T> class OpenCLSparseMatrix {
public:
  OpenCLSparseMatrix(int r, int l, int c, int s, std::vector<int> ixs,
                     std::vector<T> vals)
      : rows(r), rowlen(l), chunksize(c), splitsize(s), indices(ixs),
        values(vals){};
  // OpenCLSparseMatrix();

  int getCLVHeight() { return rows / chunksize; }
  int getCLVWidth() { return rowlen / splitsize; }
  const int rows;
  const int rowlen;
  const int chunksize;
  const int splitsize;
  const std::vector<int> indices;
  const std::vector<T> values;
};

template <typename EType> class SparseMatrix {
public:
  // Constructors
  SparseMatrix(std::string filename);
  // SparseMatrix(float lo, float hi, int length, int elements);

  // readers
  template <typename T> using ellpack_row = std::vector<std::pair<int, T>>;
  template <typename T> using ellpack_matrix = std::vector<ellpack_row<T>>;

  ellpack_matrix<EType> asELLPACK(void);

  template <typename T>
  using soa_ellpack_matrix =
      std::pair<std::vector<std::vector<int>>, std::vector<std::vector<T>>>;

  soa_ellpack_matrix<EType> asSOAELLPACK();

  soa_ellpack_matrix<EType> asPaddedSOAELLPACK(EType zero, int modulo = 1);

  // template ellpack_matrix<float> asFloatELLPACK();
  // ellpack_matrix<double> asDoubleELLPACK();
  // ellpack_matrix<int> asIntELLPACK();

  // getters
  int width();
  int height();
  int nonZeros();

  std::vector<std::tuple<int, int, EType>> getEntries();
  std::vector<int> getRowLengths(void);
  int getMaxRowEntries();
  int getMinRowEntries();
  int getMeanRowEntries();

  void printMatrix();

  // std::vector<Apart_Tuple> asSparseMatrix();
  // std::vector<Apart_Tuple> asPaddedSparseMatrix(int modulo_pad = 1);
  // std::vector<Apart_Tuple> as1DVector();

  // std::vector<int> rowLengths(void);
  // int maxRowLength();
  // skelcl::Vector<Apart_Tuple> asELLPACK(void);

private:
  // private initialisers
  void load_from_file(std::string filename);
  // void from_random_vector(float lo, float hi, int length, int elements);

  // tuples are: x, y, value
  std::vector<std::tuple<int, int, EType>> nz_entries;
  int rows;
  int cols;
  int nonz;

  std::string filename;
  std::vector<int> row_lengths;
  int max_row_entries = -1;
  int min_row_entries = -1;
  int mean_row_entries = -1;
};

#endif