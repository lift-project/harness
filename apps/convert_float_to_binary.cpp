#include <iostream>
#include <vector>

#include "file_utils.h"

using namespace std;

template <typename T>
vector<T> read_file(const string &filename) {

  ifstream in(filename);

  if (!in.good())
    throw invalid_argument("Problem with input file!");

  vector<T> contents {};

  T temp;
  while (in >> temp) {
    contents.push_back(temp);
  }

  return contents;
}

template<typename T>
void read_and_save(const string& input_filename) {
  auto output_filename = input_filename + ".binary";
  auto input = read_file<float>(input_filename);

  cout << "# of elements: " << input.size() << endl;

  for (auto& i : input)
    cout << i << " ";
  cout << endl;

  File::save_input(input, output_filename);
}

int main(int argc, char* argv[]) {

  if (argc < 2) {
    cout << "Missing input filename" << endl;
    exit(1);
  }

  string input_filename = argv[1];

  if (input_filename.find("float") != string::npos)
    read_and_save<float>(input_filename);
  else if (input_filename.find("double") != string::npos)
    read_and_save<double>(input_filename);
  else if (input_filename.find("int") != string::npos)
    read_and_save<int>(input_filename);
  else if (input_filename.find("long") != string::npos)
    read_and_save<long>(input_filename);
  else
    throw invalid_argument("Unknown output type!");
}
