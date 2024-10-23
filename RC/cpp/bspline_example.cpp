// pangolin_bspline_example.cpp
#include <iostream>
#include <memory>
#include <pangolin/pangolin.h>
#include "sim/BsplineSE3.h"
#include <Eigen/Dense>

// Helper function to draw the coordinate frame
void drawCoordinateFrame(const Eigen::Matrix4d& pose) {
    glPushMatrix();
    glMultMatrixd(pose.data());

    // Draw the x-axis (red)
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.3, 0.0, 0.0);
    glEnd();

    // Draw the y-axis (green)
    glColor3f(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0.3, 0.0);
    glEnd();

    // Draw the z-axis (blue)
    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, 0.3);
    glEnd();

    glPopMatrix();
}

int main() {
    // Initialize Pangolin window
    pangolin::CreateWindowAndBind("B-Spline SE(3) Visualization", 640, 480);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 500, 500, 320, 240, 0.2, 100),
        pangolin::ModelViewLookAt(-2, -2, 2, 0, 0, 0, pangolin::AxisY)
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // Define more complex control points for B-spline in SE(3)
    std::vector<Eigen::VectorXd> control_points;
    for (int i = 0; i < 10; ++i) {
        double time = static_cast<double>(i);  // Time for each control point
        
        // Complex translation
        double p_x = 2.0 * sin(0.5 * time);
        double p_y = 2.0 * cos(0.5 * time);
        double p_z = 0.5 * time;
        
        // Complex rotation: Creating quaternion from Euler angles
        double roll = 0.1 * time;
        double pitch = 0.2 * time;
        double yaw = 0.3 * time;

        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

        Eigen::Quaterniond quaternion = yawAngle * pitchAngle * rollAngle;
        
        // Create the control point vector: [time, p_x, p_y, p_z, q_x, q_y, q_z, q_w]
        Eigen::VectorXd control_point(8);
        control_point << time, p_x, p_y, p_z, quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w();
        control_points.push_back(control_point);
    }

    // Create and set up the B-spline instance
    std::shared_ptr<ov_core::BsplineSE3> pBSpline(new ov_core::BsplineSE3());
    pBSpline->feed_trajectory(control_points);

    // Main visualization loop
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        // Evaluate the B-spline at various time points
        std::vector<double> evaluation_times = {1.0, 2.0, 3.5, 5.5, 7.5, 9.0};
        for (double t : evaluation_times) {
            Eigen::Matrix3d R_GtoI;
            Eigen::Vector3d p_IinG;
            bool is_succeed = pBSpline->get_pose(t, R_GtoI, p_IinG);
            
            if (is_succeed) {
                // Combine rotation and translation into a 4x4 transformation matrix
                Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                pose.block<3, 3>(0, 0) = R_GtoI;
                pose.block<3, 1>(0, 3) = p_IinG;

                // Draw the coordinate frame at the evaluated pose
                drawCoordinateFrame(pose);
            }
        }

        pangolin::FinishFrame();
    }

    return 0;
}
