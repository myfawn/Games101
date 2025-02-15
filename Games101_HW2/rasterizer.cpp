// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


//generate a function that can calculate the cross product of the matrix
static auto crossValue(Eigen::Vector2f& M, Eigen::Vector2f& N)
{
    return M[0] * N[1] - M[1] * N[0];
}

static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    // get all value of the vertex of the triangle
    float A_x = _v[0][0];
    float A_y = _v[0][1];
    float B_x = _v[1][0];
    float B_y = _v[1][1];
    float C_x = _v[2][0];
    float C_y = _v[2][1];
    Eigen::Vector2f AB(B_x - A_x, B_y - A_y);
    Eigen::Vector2f BC(C_x - B_x, C_y - B_y);
    Eigen::Vector2f CA(A_x - C_x, A_y - C_y);
    Eigen::Vector2f AN(x - A_x, y - A_y);       // N is where the new point located
    Eigen::Vector2f BN(x - B_x, y - B_y);
    Eigen::Vector2f CN(x - C_x, y - C_y);

    //start the mapping returning the T or F
    if ((crossValue(AB,AN) >= 0) && (crossValue(BC, BN) >= 0) && (crossValue(CA, CN) >= 0))
    {
        return true;
    }
    else 
    {
        return false;
    }
}


static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)                                                 //will iterate for two different triangles
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
            //std::cout << vert.x() << "\n" << std::endl;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    // ****
    // Do the bounding box here, get the max and min info of the pivot, and calculate whether it is in the boundary
    // B means bottom, L means left
    int BL_x = floor(std::min(v[0].x(), std::min(v[1].x(), v[2].x())));              // note: as cpp not support 3 argu evaluation, two function is needed
    int BL_y = floor(std::min(v[0].y(), std::min(v[1].y(), v[2].y())));             //note: based on the coordinate, the origina starts in bottom left, 
    int TR_x = floor(std::max(v[0].x(), std::max(v[1].x(), v[2].x())));             //but this is not important
    int TR_y = floor(std::max(v[0].y(), std::max(v[1].y(), v[2].y())));
    
    // iterate through the pixel and find if the current pixel is inside the triangle
    // the iteration will based on the max and min value calculated above
    float pixel_x;
    float pixel_y;
    bool MASS = true;
    for (pixel_x = BL_x; pixel_x <= TR_x; ++pixel_x)
    {
        for (pixel_y = BL_y; pixel_y <= TR_y; ++pixel_y)
        {
            if (MASS)       //using MASS
            {
                const float axis_x[4] = { +0.25, +0.75, +0.25, +0.75 };
                const float axis_y[4] = { +0.25, +0.25, +0.75, +0.75 };
                for (int i = 0; i < 4; ++i)
                {
                    int samplePid = get_index(pixel_x, pixel_y) * 4 + i;                    // as there are 4 times original pixel, we need to use this function to get a personalized id
                    if (insideTriangle(pixel_x + axis_x[i], pixel_y + axis_y[i], t.v))          // this statement will check if the 2x2 point is in the triangle or not
                    {
                        auto [alpha, beta, gamma] = computeBarycentric2D(pixel_x + axis_x[i], pixel_y + axis_y[i], t.v);            //These variables will be used to find the corresponding value of the RGB based on different vertex
                        float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                        z_interpolated *= w_reciprocal;         //this is the calculated depth for the specific pixel based on index 1-4
                        // std::cout<<"1: "<< z_interpolated <<std::endl;
                        //std::cout<<"2: "<< sample_frame_buf[samplePid] <<std::endl;
                        //std::cout<<"color: "<< t.getColor() <<std::endl;
                        if (-z_interpolated < sample_depth_buf[samplePid])
                        {
                            sample_depth_buf[samplePid] = -z_interpolated;
                            sample_frame_buf[samplePid] = t.getColor() / 4.0;
                            depth_buf[get_index(pixel_x, pixel_y)] = std::min(depth_buf[get_index(pixel_x, pixel_y)], -z_interpolated);      //find the closest obj
                            Eigen::Vector3f setColor = sample_frame_buf[get_index(pixel_x, pixel_y) * 4] + sample_frame_buf[get_index(pixel_x, pixel_y) * 4 + 1]+ sample_frame_buf[get_index(pixel_x, pixel_y) * 4+2]+ sample_frame_buf[get_index(pixel_x, pixel_y) * 4 + 3];
                            // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                            set_pixel(Eigen::Vector3f(pixel_x, pixel_y, depth_buf[get_index(pixel_x, pixel_y)]), setColor);
                        }
                    }
                }
            }
            else            //Not using MASS
            {
                if (insideTriangle(pixel_x + 0.5, pixel_y + 0.5, t.v))
                {
                    // If so, use the following code to get the interpolated z value.
                    auto [alpha, beta, gamma] = computeBarycentric2D(pixel_x, pixel_y, t.v);            //These variables will be used to find the corresponding value of the RGB based on different vertex
                    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;         //this is the calculated depth for the specific pixel
                    //float temp_depth = depth_buf[get_index(pixel_x, pixel_y)];
                    //std::cout<<"z: "<< get_index(pixel_x, pixel_y) <<std::endl;
                    //std::cout<<"v[0].z(): "<<v[0].z()<<std::endl;
                    //z_interpolated = std::fabs(z_interpolated);
                    if (-z_interpolated < depth_buf[get_index(pixel_x, pixel_y)]) 
                    {
                        depth_buf[get_index(pixel_x, pixel_y)] = -z_interpolated;
                        // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                        set_pixel(Eigen::Vector3f(pixel_x, pixel_y, z_interpolated), t.getColor());
                    }
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});                    // initially should be dark for each color
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());      // initially it is finifite far from the screen
    }

    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(sample_frame_buf.begin(), sample_frame_buf.end(), Eigen::Vector3f{ 0, 0, 0 });                    // initially should be dark for each color
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(sample_depth_buf.begin(), sample_depth_buf.end(), std::numeric_limits<float>::infinity());      // initially it is finifite far from the screen
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);

    sample_depth_buf.resize(w * h * 4);
    //std::fill(sample_depth_buf.begin(), sample_depth_buf.end(), std::numeric_limits<float>::infinity());
    sample_frame_buf.resize(w * h * 4);
    //std::fill(sample_frame_buf.begin(), sample_frame_buf.end(), Eigen::Vector3f{ 0, 0, 0 });
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)         // this function will color the point
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on