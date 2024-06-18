#include "Triangle.hpp"
#include "rasterizer.hpp"
//#include "Eigen/Eigen"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.

    // the rotational matrix should be [cos_theta -sin_theta ; sin_theta cos_theta]
    // Note: a 4x4 matrix is needed, but no translation needed
    Eigen::Matrix4f rotationMatrix;
    rotationMatrix << std::cos(rotation_angle), -std::sin(rotation_angle), 0, 0,
        std::sin(rotation_angle), std::cos(rotation_angle), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 0;

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

    Eigen::Matrix4f perspToOrthoMatrix = Eigen::Matrix4f::Zero();
    perspToOrthoMatrix(0, 0) = zNear;
    perspToOrthoMatrix(1, 1) = zNear;
    perspToOrthoMatrix(2, 2) = zNear + zFar;
    perspToOrthoMatrix(2, 3) = -zFar * zNear;
    perspToOrthoMatrix(3, 2) = 1.0f;

    float t = std::tan(eye_fov / 2.0f) * zNear;
    float b = -t;
    float r = t * aspect_ratio;
    float l = -r;

    Eigen::Matrix4f orthoMatrix = Eigen::Matrix4f::Zero();
    orthoMatrix(0, 0) = 2.0f / (r - l);
    orthoMatrix(1, 1) = 2.0f / (t - b);
    orthoMatrix(2, 2) = 2.0f / (zNear - zFar);
    orthoMatrix(0, 3) = -(r + l) / (r - l);
    orthoMatrix(1, 3) = -(t + b) / (t - b);
    orthoMatrix(2, 3) = -(zNear + zFar) / (zNear - zFar);
    orthoMatrix(3, 3) = 1.0f;

    projection = orthoMatrix * perspToOrthoMatrix * projection;

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
