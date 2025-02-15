//
// Created by goksu on 4/6/19.
//

#include <algorithm>
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

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f>& normals)
{
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}


// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x,y,dx,dy,dx1,dy1,px,py,xe,ye,i;

    dx=x2-x1;
    dy=y2-y1;
    dx1=fabs(dx);
    dy1=fabs(dy);
    px=2*dy1-dx1;
    py=2*dx1-dy1;

    if(dy1<=dx1)
    {
        if(dx>=0)
        {
            x=x1;
            y=y1;
            xe=x2;
        }
        else
        {
            x=x2;
            y=y2;
            xe=x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;x<xe;i++)
        {
            x=x+1;
            if(px<0)
            {
                px=px+2*dy1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    y=y+1;
                }
                else
                {
                    y=y-1;
                }
                px=px+2*(dy1-dx1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
    else
    {
        if(dy>=0)
        {
            x=x1;
            y=y1;
            ye=y2;
        }
        else
        {
            x=x2;
            y=y2;
            ye=y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;y<ye;i++)
        {
            y=y+1;
            if(py<=0)
            {
                py=py+2*dx1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    x=x+1;
                }
                else
                {
                    x=x-1;
                }
                py=py+2*(dx1-dy1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(int x, int y, const Vector4f* _v){
    Vector3f v[3];
    for(int i=0;i<3;i++)
        v[i] = {_v[i].x(),_v[i].y(), 1.0};
    Vector3f f0,f1,f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x,y,1.);
    if((p.dot(f0)*f0.dot(v[2])>0) && (p.dot(f1)*f1.dot(v[0])>0) && (p.dot(f2)*f2.dot(v[1])>0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v){
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList) {

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    // for each triangle do the operation
    for (const auto& t:TriangleList)
    {
        Triangle newtri = *t;

        //mm will get all info for the triangle and do the MV operation
        std::array<Eigen::Vector4f, 3> mm {
                (view * model * t->v[0]),       // normal
                (view * model * t->v[1]),       // texture
                (view * model * t->v[2])        // vertex
        };

        // find view space position
        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto& v) {
            return v.template head<3>();
        });

        // do the mvp for all points in v
        Eigen::Vector4f v[] = {
                mvp * t->v[0],
                mvp * t->v[1],
                mvp * t->v[2]
        };
        //Homogeneous division
        // mainly used for transformation
        for (auto& vec : v) {
            vec.x()/=vec.w();
            vec.y()/=vec.w();
            vec.z()/=vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        // after MV transformation, the normal line value is now directed to the eye
        Eigen::Vector4f n[] = {
                inv_trans * to_vec4(t->normal[0], 0.0f),
                inv_trans * to_vec4(t->normal[1], 0.0f),
                inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        //Viewport transformation
        // this is where the camera located, will find the location for each triangle
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        // set the vertex of each point in triangle
        for (int i = 0; i < 3; ++i)
        {
            //screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        // set the normal on each vertex of the triangle
        for (int i = 0; i < 3; ++i)
        {
            //view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        // force the init color
        newtri.setColor(0, 148,121.0,92.0);
        newtri.setColor(1, 148,121.0,92.0);
        newtri.setColor(2, 148,121.0,92.0);

        // Also pass view space vertice position
        rasterize_triangle(newtri, viewspace_pos);
    }
}

static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f& vert1, const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3, float weight)
{
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f& vert1, const Eigen::Vector2f& vert2, const Eigen::Vector2f& vert3, float weight)
{
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos) 
{
    // TODO: From your HW3, get the triangle rasterization code.
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    // ****
    // Do the bounding box here, get the max and min info of the pivot, and calculate whether it is in the boundary
    // B means bottom, L means left
    int BL_x = floor(std::min(v[0].x(), std::min(v[1].x(), v[2].x())));              // note: as cpp not support 3 argu evaluation, two function is needed
    int BL_y = floor(std::min(v[0].y(), std::min(v[1].y(), v[2].y())));             //note: based on the coordinate, the origina starts in bottom left, 
    int TR_x = ceil(std::max(v[0].x(), std::max(v[1].x(), v[2].x())));             //but this is not important
    int TR_y = ceil(std::max(v[0].y(), std::max(v[1].y(), v[2].y())));

    // iterate through the pixel and find if the current pixel is inside the triangle
    // the iteration will based on the max and min value calculated above
    float pixel_x;
    float pixel_y;
    bool MASS = false;
    for (pixel_x = BL_x; pixel_x <= TR_x; ++pixel_x)
    {
        for (pixel_y = BL_y; pixel_y <= TR_y; ++pixel_y)
        {
            if (MASS)       //using MASS
            {
                //const float axis_x[4] = { +0.25, +0.75, +0.25, +0.75 };
                //const float axis_y[4] = { +0.25, +0.25, +0.75, +0.75 };
                //for (int i = 0; i < 4; ++i)
                //{
                //    int samplePid = get_index(pixel_x, pixel_y) * 4 + i;                    // as there are 4 times original pixel, we need to use this function to get a personalized id
                //    if (insideTriangle(pixel_x + axis_x[i], pixel_y + axis_y[i], t.v))          // this statement will check if the 2x2 point is in the triangle or not
                //    {
                //        auto [alpha, beta, gamma] = computeBarycentric2D(pixel_x + axis_x[i], pixel_y + axis_y[i], t.v);            //These variables will be used to find the corresponding value of the RGB based on different vertex
                //        float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                //        float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                //        z_interpolated *= w_reciprocal;         //this is the calculated depth for the specific pixel based on index 1-4
                //        // std::cout<<"1: "<< z_interpolated <<std::endl;
                //        //std::cout<<"2: "<< sample_frame_buf[samplePid] <<std::endl;
                //        //std::cout<<"color: "<< t.getColor() <<std::endl;
                //        if (-z_interpolated < sample_depth_buf[samplePid])
                //        {
                //            sample_depth_buf[samplePid] = -z_interpolated;
                //            sample_frame_buf[samplePid] = t.getColor() / 4.0;
                //            depth_buf[get_index(pixel_x, pixel_y)] = std::min(depth_buf[get_index(pixel_x, pixel_y)], -z_interpolated);      //find the closest obj
                //            Eigen::Vector3f setColor = sample_frame_buf[get_index(pixel_x, pixel_y) * 4] + sample_frame_buf[get_index(pixel_x, pixel_y) * 4 + 1] + sample_frame_buf[get_index(pixel_x, pixel_y) * 4 + 2] + sample_frame_buf[get_index(pixel_x, pixel_y) * 4 + 3];
                //            // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                //            set_pixel(Eigen::Vector3f(pixel_x, pixel_y, depth_buf[get_index(pixel_x, pixel_y)]), setColor);
                //        }
                //    }
                //}
            }
            else            //Not using MASS
            {
                // TODO: Inside your rasterization loop:
                //    * v[i].w() is the vertex view space depth value z.
                //    * Z is interpolated view space depth for the current pixel
                //    * zp is depth between zNear and zFar, used for z-buffer

                // float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                // float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                // zp *= Z;
                if (insideTriangle(pixel_x + 0.5, pixel_y + 0.5, t.v))
                {
                    // If so, use the following code to get the interpolated z value.
                    auto [alpha, beta, gamma] = computeBarycentric2D(pixel_x, pixel_y, t.v);            //These variables will be used to find the corresponding value of the RGB based on different vertex
                    float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    zp *= Z;         //this is the calculated depth for the specific pixel
                    float temp_depth = depth_buf[get_index(pixel_x, pixel_y)];
                    //std::cout<<"z: "<< get_index(pixel_x, pixel_y) <<std::endl;
                    //std::cout<<"v[0].z(): "<<v[0].z()<<std::endl;
                    //z_interpolated = std::fabs(z_interpolated);
                    if (zp < depth_buf[get_index(pixel_x, pixel_y)])
                    {
                        // TODO: Interpolate the attributes:
                        // based on the interpolate for each triangle, we can get and approximate the property of each triangle including texture, color, etc.
                        // interpolate is similar to the depth?
                        // note: this code is using 2D for calculating the 3D example, in reality, we need to use zp and other variables to make the result more accurate
                        // find the color to be shaded
                        auto interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1);
                        // find the normal of the triangle
                        auto interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1);
                        // texture coordinate, this equation will get the location of the coordinate based on the mapping
                        auto interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1);
                        // shading coordinate, similar to the position to apply the shading
                        auto interpolated_shadingcoords = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], 1);
                        fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
                        payload.view_pos = interpolated_shadingcoords;
                        //Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
                        auto pixel_color = fragment_shader(payload);
                        //set_pixel(Vector2i(pixel_x, pixel_y), pixel_color);
                        // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                        depth_buf[get_index(pixel_x, pixel_y)] = zp;
                        set_pixel(Vector2i(pixel_x, pixel_y), pixel_color);

                        //example
                        //auto interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1);
                        //auto interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1);
                        //auto interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1);
                        //auto interpolated_shadingcoords = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], 1);
                        //fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
                        //payload.view_pos = interpolated_shadingcoords;
                        //depth_buf[get_index(pixel_x, pixel_y)] = zp;
                        //auto pixel_color = fragment_shader(payload);
                        //set_pixel(Vector2i(pixel_x, pixel_y), pixel_color);
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
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
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
    sample_frame_buf.resize(w * h * 4);

    texture = std::nullopt;
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-y)*width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
{
    //old index: auto ind = point.y() + point.x() * width;
    int ind = (height-point.y())*width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}

