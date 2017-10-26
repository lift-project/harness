#include <iostream>
#include <vector>

#include "file_utils.h"

using namespace std;

template <typename T>
vector<T> read_file(const string &filename) {

  vector<T> contents {};
  ifstream in(filename);

  if (!in.good())
    return contents;

  while (!in.eof()) {
    T temp;
    in >> temp;
    contents.push_back(temp);
  }

  return contents;
}

int main(int argc, char* argv[]) {

  if (argc < 2) {
    cout << "Missing input filename" << endl;
    exit(1);
  }

  string input_filename = argv[1];
  auto output_filename = input_filename + ".binary";

  auto input = read_file<float>(input_filename);

  cout << "# of elements: " << input.size() << endl;

  File::save_input(input, output_filename);
  return 0;
}
