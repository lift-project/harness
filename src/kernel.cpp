#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "file_utils.h"

#include "kernel.h"

Kernel::Kernel(std::string filename) {
  boost::property_tree::ptree tree;

  boost::property_tree::read_json(filename, tree);

  name = tree.get<std::string>("name");
  source = tree.get<std::string>("source");

  std::cout << "Kernel: " << name << ", source: \n" << source << std::endl;

  for (auto &arg : tree.get_child("args")) {
    std::string variable = arg.second.get<std::string>("variable");
    std::string addressSpace = arg.second.get<std::string>("addressSpace");
    std::string size = arg.second.get<std::string>("size");
    args.push_back(KernelArg(variable, addressSpace, size));

    std::cout << "variable: " << variable << " address space: " << addressSpace
              << " size: " << size << std::endl;
  }
}

std::string Kernel::getSource() { return source; }

std::string Kernel::getName() { return name; }

std::vector<KernelArg> Kernel::getArgs() { return args; }