#include <iostream>

#include "reader.hpp"
#include "data.hpp"
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
  reader->Load(data_path);

  // Get data from reader
  Data *data = reader->GetData();

  Settings *settings = new Settings();


  // settings for forest
  settings->num_trees_ = 5;
  settings->max_tree_depth_ = 16;

  // other settings
  int num_frames_per_tree = 500;
  int samples_per_frame = 5000;

  // Create forest
  Forest *forest = new Forest(data, settings);

  // train forest
  forest->Train();

  // test forest with random data 

  return 0;
}