#include <iostream>

#include "reader.hpp"
#include "settings.hpp"
#include "forest.hpp"

using namespace std;
using namespace ISUE::RelocForests;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: ./relocforests <path_to_association_file>";
    return 1;
  }

  // get path
  string data_path(argv[1]);

  Reader *reader = new Reader();
  bool err = reader->Load(data_path);
  if (err) {
    return 1;
  }

  // Get data from reader
  Data *data = reader->GetData();

  // settings for forest
  Settings *settings = new Settings();
  settings->fx = 525.0f;
  settings->fy = 525.0f;
  settings->cx = 319.5f;
  settings->cy = 239.5f;

  // Create forest
  Forest *forest = new Forest(data, *settings);

  // train forest
  forest->Train();

  // test forest with random data

  cout << "Done Training.\n";

  delete forest;
  delete reader;
  delete settings;


  return 0;
}
