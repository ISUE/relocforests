/*
 * MeanShift implementation created by Matt Nedrich.
 * project: https://github.com/mattnedrich/MeanShift_cpp
 *
 */
#include <vector>
#include <bitset>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

class MeanShift {
public:
    MeanShift() { set_kernel(NULL); }
    MeanShift(double (*_kernel_func)(double,double)) { set_kernel(kernel_func); }
    vector<Vector3d> cluster(vector<Vector3d>, double);
private:
    double (*kernel_func)(double,double);
    void set_kernel(double (*_kernel_func)(double,double));
    Vector3d shift_point(const Vector3d &, const vector<Vector3d> &, double);
};
