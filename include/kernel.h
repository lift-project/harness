#ifndef KERNEL_H
#define KERNEL_H

#include "sparse_matrix.hpp"

class KernelArg {
public:
  const std::string variable;
  const std::string addressSpace;
  const std::string size;
  KernelArg(std::string var, std::string aspce, std::string sz)
      : variable(var), addressSpace(aspce), size(sz){};
};

// ignore this for now - let's get stuff working for simple kernels first
class KernelProperties {
public:
  // Constructor
  KernelProperties();
  KernelProperties(std::string kname);

private:
  std::string argcache;
};

// template <typename T>
class Kernel {
public:
  // Constructors
  Kernel(std::string filename);

  // Destuctor
  ~Kernel(){};

  // Getters
  std::string getSource();
  std::string getName();
  std::vector<KernelArg> getArgs();
  KernelProperties getProperties();

  // Specialiser for a matrix - makes more sense here than in the matrix,
  // as it's kernel, not matrix dependent

  // OpenCLSparseMatrix<T> specialiseMatrix(SparseMatrix matrix);

private:
  std::string source;
  std::string name;
  std::vector<KernelArg> args;
  KernelProperties kprops;
};

#endif // KERNEL_H