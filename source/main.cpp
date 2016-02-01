#include <iostream>

#include "reader.hpp"
#include "settings.hpp"
#include "forest.hpp"

using namespace std;
using namespace ISUE::RelocForests;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: [train||test] ./relocforests <path_to_association_file>";
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

  Forest *forest = nullptr;

  bool train = false;
  if (train) {
    forest = new Forest(data, settings);
    forest->Train();
    forest->Serialize("forest.rf");

    cout << "Is forest valid:" << forest->IsValid() << endl;
  }
  else {

    // load forest
    forest = new Forest(data, settings, "forest.rf");

    cout << "Is forest valid:" << forest->IsValid() << endl;

    // eval forest at frame
    std::clock_t start;
    double duration;
    start = std::clock();

    Eigen::Affine3d pose = forest->Test(data->GetRGBImage(200), data->GetDepthImage(200));

    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    cout << "Train time: " << duration << " Seconds \n";

    // compare pose to known value 
    auto known_pose = data->poses_eigen_.at(200);

    cout << "found pose:" << endl;
    cout << pose.rotation() << endl << endl;
    cout << pose.rotation().eulerAngles(0, 1, 2) * 180 / M_PI << endl;
    cout << pose.translation() << endl;

    cout << "known pose:" << endl;
    cout << known_pose.first << endl;
    cout << known_pose.first.eulerAngles(0, 1, 2) * 180 / M_PI << endl;
    cout << known_pose.second << endl;

    if ((known_pose.first - pose.rotation()).cwiseAbs().maxCoeff() > 1e-13 ||
      (known_pose.second - pose.translation()).cwiseAbs().maxCoeff() > 1e-13)
      cout << "Pose could not be found\n";
    else
      cout << "Pose was found!\n";
  }

  cout << "Done.\n";

  delete forest;
  delete reader;
  delete settings;


  return 0;
}
