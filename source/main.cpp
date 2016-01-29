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
  Forest *forest = new Forest(data, settings, "forest.rf");

  bool check = forest->IsValid();
  // train forest
  //forest->Train();
  //forest->Serialize("forest.rf");
  auto hypotheses = forest->Test(data->GetRGBImage(54), data->GetDepthImage(54));
  

  auto known_pose = data->poses_eigen_.at(54);

  for (auto h : hypotheses) {
    auto pose = h.pose;

    auto rot = pose.rotation();
    auto lin = pose.linear();
    auto trans = pose.translation();

    cout << "found pose" << endl;
    cout << rot << endl << endl;
    cout << lin << endl;
    cout << trans << endl;

    cout << "known pose" << endl;
    cout << known_pose.first << endl;
    cout << known_pose.second << endl;
    if ((known_pose.first - pose.linear()).cwiseAbs().maxCoeff() > 1e-13 ||
      (known_pose.second - pose.translation()).cwiseAbs().maxCoeff() > 1e-13)
      cout << "Pose could not be found\n";
    else
      cout << "Pose was found!\n";
  }

  // test forest with random data

  cout << "Done Training.\n";

  delete forest;
  delete reader;
  delete settings;


  return 0;
}
