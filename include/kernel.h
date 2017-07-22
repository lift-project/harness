#ifndef KERNEL_H
#define KERNEL_H

class KernelArg {
public:
  const std::string variable;
  const std::string addressSpace;
  const std::string size;
  KernelArg(std::string var, std::string aspce, std::string sz)
      : variable(var), addressSpace(aspce), size(sz){};
};

class Kernel {
public:
  // Constructors
  Kernel(std::string filename);

  // Destuctor
  ~Kernel(){};

  std::string getSource();
  std::string getName();
  std::vector<KernelArg> getArgs();

private:
  std::string source;
  std::string name;
  std::vector<KernelArg> args;
};

#endif // KERNEL_H