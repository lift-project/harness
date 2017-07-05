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

class SparseMatrix {
public:
  // Constructors
  SparseMatrix(std::string filename);
  SparseMatrix(float lo, float hi, int length, int elements);

  // readers
  template <typename EType>
  using ellpack_row = std::vector<std::pair<int, EType>>;

  template <typename EType>
  using ellpack_matrix = std::vector<ellpack_row<EType>>;

  template <typename T> ellpack_matrix<T> asELLPACK(void);

  template <typename EType>
  using soa_ellpack_matrix =
      std::pair<std::vector<std::vector<int>>, std::vector<std::vector<EType>>>;

  template <typename T> soa_ellpack_matrix<T> asSOAELLPACK();

  template <typename T>
  soa_ellpack_matrix<T> asPaddedSOAELLPACK(T zero, int modulo = 1);

  // template ellpack_matrix<float> asFloatELLPACK();
  // ellpack_matrix<double> asDoubleELLPACK();
  // ellpack_matrix<int> asIntELLPACK();

  // getters
  int width();
  int height();
  int nonZeros();

  double maxElement();
  double minElement();

  std::vector<std::tuple<int, int, double>> getEntries();
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
  void from_random_vector(float lo, float hi, int length, int elements);

  // tuples are: x, y, value
  std::vector<std::tuple<int, int, double>> nz_entries;
  int rows;
  int cols;
  int nonz;
  double ma_elem;
  double mi_elem;
  std::string filename;
  std::vector<int> row_lengths;
  int max_row_entries = -1;
  int min_row_entries = -1;
  int mean_row_entries = -1;
};

#endif