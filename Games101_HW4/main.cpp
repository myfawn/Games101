#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> & control_points, float t)       // aims to find the point waiting to connect together
{
    // TODO: Implement de Casteljau's algorithm
    auto& p_0 = control_points[0];
    auto& p_1 = control_points[1];
    auto& p_2 = control_points[2];
    auto& p_3 = control_points[3];
    double m = t / (t - 1);
    auto one_01 = p_0 + t * (p_1 - p_0);
    auto one_12 = p_1 + t * (p_2 - p_1);
    auto one_23 = p_2 + t * (p_3 - p_2);
    auto two_02 = one_01 + t * (one_12 - one_01);
    auto two_13 = one_12 + t * (one_23 - one_12);
    auto draw = two_02 + t * (two_13 - two_02);
    return cv::Point2f(draw);

}

float distance(cv::Point2f& a, cv::Point2f& b)
{
    if (a.x >= 0 && a.x < 700 && a.y >= 0 && a.y < 700) 
    {
        //std::cout << "round" << std::endl;
        //std::cout << a.x - b.x << std::endl;
        //std::cout << a.y - b.y << std::endl;
        //std::cout << std::sqrt((std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2))) << std::endl;
        return (std::sqrt((std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2))));
    }
    else
    {
        return 1;
    }

    //return (std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)));
}

void Bilinear(cv::Point2f& point, cv::Mat& window)
{
    int x_left = std::max(0.f, std::floor(point.x));
    int x_right = std::max(0.f, std::ceil(point.x));
    int y_bottom = std::max(0.f, std::floor(point.y));       //cv view
    int y_top = std::max(0.f, std::ceil(point.y));           //cv view
    ////std::cout << "Round: " << std::endl;
    ////std::cout << point.x << std::endl;
    ////std::cout << point.y << std::endl;
    ////std::cout << x_left << std::endl;
    ////std::cout << x_right << std::endl;
    ////std::cout << y_bottom << std::endl;
    ////std::cout << y_top << std::endl;
    cv::Point2f u00(x_left + 0.5, y_top + 0.5);
    cv::Point2f u01(x_left + 0.5, y_bottom + 0.5);
    cv::Point2f u10(x_right + 0.5, y_top + 0.5);
    cv::Point2f u11(x_right + 0.5, y_bottom + 0.5);
    window.at<cv::Vec3b>(u00.y, u00.x)[1] = std::min(255.f, 255 / distance(u00, point));
    window.at<cv::Vec3b>(u01.y, u01.x)[1] = std::min(255.f, 255 / distance(u01, point));
    window.at<cv::Vec3b>(u10.y, u10.x)[1] = std::min(255.f, 255 / distance(u10, point));
    window.at<cv::Vec3b>(u11.y, u11.x)[1] = std::min(255.f, 255 / distance(u11, point));
}


void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.
    for (double t = 0.0; t <= 1.0; t += 0.001)
    {
        auto point = recursive_bezier(control_points, t);
        Bilinear(point, window);
        //window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
    }
}

int main() 
{   
    //start the botton
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));         //generate a window for creating the line
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);       // start creating the control points

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4) 
        {
            //control_points = {
            //    cv::Point2f(1.f, 1.f),
            //    cv::Point2f(2.f, 2.f),
            //    cv::Point2f(3.f, 3.f),
            //    cv::Point2f(4.f, 4.f)
            //};
            //naive_bezier(control_points, window);
            bezier(control_points, window);
            //std::cout << control_points << std::endl;
            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}
