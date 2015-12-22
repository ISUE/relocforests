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

  // settings for forest
  Settings *settings = new Settings();
  settings->num_trees_ = 5;
  settings->max_tree_depth_ = 16;
  settings->num_frames_per_tree_ = 500;
  settings->num_pixels_per_frame_ = 5000;
  settings->image_width_ = 640;
  settings->image_height_ = 480;
  settings->depth_factor_ = 5000;

  settings->fx = 525.0;
  settings->fy = 525.0;
  settings->cx = 319.5;
  settings->cy = 239.5;

  // Create forest
  Forest *forest = new Forest(data, settings);

  // train forest
  forest->Train();

  // test forest with random data 

  delete forest;

  return 0;
}