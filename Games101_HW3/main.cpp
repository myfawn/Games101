#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

float h(float u, float v, const fragment_shader_payload& payload)
{
    return payload.texture->getColor(u, v).norm();
}

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Use the same projection matrix from the previous assignments
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

    float t = tan((eye_fov * MY_PI / 180) / 2) * fabs(zNear);
    //float t = std::tan(eye_fov / 2.0f) * zNear;
    float b = -t;
    float r = t * aspect_ratio;
    float l = -r;

    Eigen::Matrix4f orthoMatrix = Eigen::Matrix4f::Identity();
    orthoMatrix(0, 0) = 2.0f / (r - l);
    orthoMatrix(1, 1) = 2.0f / (t - b);
    orthoMatrix(2, 2) = 2.0f / (zNear - zFar);
    orthoMatrix(0, 3) = -(r + l) / (r - l);
    orthoMatrix(1, 3) = -(t + b) / (t - b);
    orthoMatrix(2, 3) = -(zNear + zFar) / (zNear - zFar);

    projection = orthoMatrix * perspToOrthoMatrix * projection;

    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

// this function will return the color based on the normal, the color will be lighter if the normal is directed to the light
Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    // range originally [-1,1], then back to [0,1] after the plus sign
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    Eigen::Vector3f pink_color(255, 170, 193);
    float blend_factor = 0.0;      //0.75
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    result = blend_factor * pink_color + (1.f - blend_factor) * result;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        float u = std::max(0.f, payload.tex_coords[0]);
        float v = std::max(0.f, payload.tex_coords[1]);
        return_color = payload.texture->getColor(u, v);
        auto temp = payload.texture;
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    // start edit
    Eigen::Vector3f L_ambient = { 0, 0, 0 };
    Eigen::Vector3f L_diffuse = { 0, 0, 0 };
    Eigen::Vector3f L_specular = { 0, 0, 0 };
    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        // local variable
        Eigen::Vector3f direction_point2light = light.position - point;
        float r_square = direction_point2light.dot(direction_point2light);
        Eigen::Vector3f light_energy2point = light.intensity / r_square;
        Eigen::Vector3f direction_point2eye = eye_pos - point;
        //Eigen::Vector3f light_energy2eye = light.intensity / r_square;

        //// find the increament *ambient*
        L_ambient += ka.cwiseProduct(amb_light_intensity);     //Note:I_a is a constant in this case

        // find the increament *diffuse*
        float diffuse_max = std::max(0.f, normal.dot(direction_point2light.normalized()));
        L_diffuse += kd.cwiseProduct(light_energy2point) * diffuse_max;

        // find the increament *specular*
        Eigen::Vector3f h_sum_specular = direction_point2light + direction_point2eye;
        float magnitude_specular = h_sum_specular.norm();
        float specular_max = std::max(0.f, normal.dot(h_sum_specular / magnitude_specular));
        float specular_max_tempVector = std::pow(specular_max, p);
        L_specular += ks.cwiseProduct(light_energy2point) * specular_max_tempVector;
    }
    result_color = L_ambient + L_diffuse + L_specular;
    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    // the light recorded the location and the intensity for the light, 500 represent the intensity for the RGB color respectively
    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    // point is the point where shading is going to happen
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    // start edit
    Eigen::Vector3f L_ambient = {0, 0, 0 };
    Eigen::Vector3f L_diffuse = {0, 0, 0 };
    Eigen::Vector3f L_specular = {0, 0, 0 };
    for (auto& light : lights)      // this is the loop for two light source looping through all triangles; note: only two light ray for each triangle
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        // local variable
        Eigen::Vector3f direction_point2light = light.position - point;
        float r_square = direction_point2light.dot(direction_point2light);
        Eigen::Vector3f light_energy2point = light.intensity / r_square;
        Eigen::Vector3f direction_point2eye = eye_pos - point;
        //Eigen::Vector3f light_energy2eye = light.intensity / r_square;

        
        //// find the increament *ambient*
        L_ambient +=  ka.cwiseProduct(amb_light_intensity);     //Note:I_a is a constant in this case
        
        // find the increament *diffuse*
        float diffuse_max = std::max(0.f, normal.dot(direction_point2light.normalized()));
        L_diffuse += kd.cwiseProduct(light_energy2point) * diffuse_max;
        
        // find the increament *specular*
        Eigen::Vector3f h_sum_specular = direction_point2light + direction_point2eye;
        float magnitude_specular = h_sum_specular.norm();
        float specular_max = std::max(0.f, normal.dot(h_sum_specular / magnitude_specular));
        float specular_max_tempVector = std::pow(specular_max, p);
        L_specular += ks.cwiseProduct(light_energy2point) * specular_max_tempVector;

        //example
        //// 光的方向
        //Eigen::Vector3f light_dir = light.position - point;
        //// 视线方向
        //Eigen::Vector3f view_dir = eye_pos - point;
        //// 衰减因子
        //float r = light_dir.dot(light_dir);
        //
        //// ambient
        //Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);
        // diffuse
        //Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / r);
        //Ld *= std::max(0.0f, normal.normalized().dot(light_dir.normalized()));
        //// specular
        //Eigen::Vector3f h = (light_dir + view_dir).normalized();
        //Eigen::Vector3f Ls = ks.cwiseProduct(light.intensity / r);
        //Ls *= std::pow(std::max(0.0f, normal.normalized().dot(h)), p);
        //
        //result_color += (La + Ld + Ls);  
    }
    result_color = L_ambient + L_diffuse + L_specular;
    return result_color * 255.f;
}



Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
        
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)

    //start
    Eigen::Vector3f n = normal;
    Eigen::Vector3f t = { n.x() * n.y() / std::sqrt(n.x() * n.x() + n.z() * n.z()),
                            std::sqrt(n.x() * n.x() + n.z() * n.z()),
                            n.z() * n.y() / sqrt(n.x() * n.x() + n.z() * n.z()) };
    Eigen::Vector3f b = n.cwiseProduct(t);

    Eigen::Matrix3f TBN;
    TBN.col(0) = t;
    TBN.col(1) = b;
    TBN.col(2) = n;

    float u;
    float v;
    float w = payload.texture->height;
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        u = std::max(0.f, payload.tex_coords[0]);
        v = std::max(0.f, payload.tex_coords[1]);
    }

    auto dU = kh * kn * (h(u + 1 / w, v, payload) - h(u, v, payload));
    auto dV = kh * kn * (h(u, v + 1 / w, payload) - h(u, v, payload));

    Eigen::Vector3f ln = { -dU, -dV, 1 };
    point += kn * n * h(u, v, payload);
    normal = (TBN * ln).normalized();


    Eigen::Vector3f result_color = {0, 0, 0};

    // start edit
    Eigen::Vector3f L_ambient = { 0, 0, 0 };
    Eigen::Vector3f L_diffuse = { 0, 0, 0 };
    Eigen::Vector3f L_specular = { 0, 0, 0 };

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        // local variable
        Eigen::Vector3f direction_point2light = light.position - point;
        float r_square = direction_point2light.dot(direction_point2light);
        Eigen::Vector3f light_energy2point = light.intensity / r_square;
        Eigen::Vector3f direction_point2eye = eye_pos - point;
        //Eigen::Vector3f light_energy2eye = light.intensity / r_square;


        //// find the increament *ambient*
        L_ambient += ka.cwiseProduct(amb_light_intensity);     //Note:I_a is a constant in this case

        // find the increament *diffuse*
        float diffuse_max = std::max(0.f, normal.dot(direction_point2light.normalized()));
        L_diffuse += kd.cwiseProduct(light_energy2point) * diffuse_max;

        // find the increament *specular*
        Eigen::Vector3f h_sum_specular = direction_point2light + direction_point2eye;
        float magnitude_specular = h_sum_specular.norm();
        float specular_max = std::max(0.f, normal.dot(h_sum_specular / magnitude_specular));
        float specular_max_tempVector = std::pow(specular_max, p);
        L_specular += ks.cwiseProduct(light_energy2point) * specular_max_tempVector;
    }
    result_color = L_ambient + L_diffuse + L_specular;
    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)

    //start
    Eigen::Vector3f n = normal;
    Eigen::Vector3f t = {n.x() * n.y() / std::sqrt(n.x() * n.x() + n.z() * n.z()), 
                            std::sqrt(n.x() * n.x() + n.z() * n.z()), 
                            n.z()* n.y() / sqrt(n.x() * n.x() + n.z() * n.z())};
    Eigen::Vector3f b = n.cwiseProduct(t);
    
    Eigen::Matrix3f TBN;
    TBN.col(0) = t;
    TBN.col(1) = b;
    TBN.col(2) = n;
    
    float u;
    float v;
    float w = payload.texture->height;
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        u = std::max(0.f, payload.tex_coords[0]);
        v = std::max(0.f, payload.tex_coords[1]);
    }
    
    auto dU = kh * kn * (h(u + 1 / w, v, payload) - h(u, v, payload));
    auto dV = kh * kn * (h(u, v + 1 / w, payload) - h(u, v, payload));
    
    Eigen::Vector3f ln = { -dU, -dV, 1 };
    normal = (TBN * ln).normalized();
    
    Eigen::Vector3f result_color = {0, 0, 0};

    // start edit
    //Eigen::Vector3f L_ambient = { 0, 0, 0 };
    //Eigen::Vector3f L_diffuse = { 0, 0, 0 };
    //Eigen::Vector3f L_specular = { 0, 0, 0 };
    //for (auto& light : lights)
    //{
    //    // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
    //    // components are. Then, accumulate that result on the *result_color* object.
    //    // local variable
    //    Eigen::Vector3f direction_point2light = light.position - point;
    //    float r_square = direction_point2light.dot(direction_point2light);
    //    Eigen::Vector3f light_energy2point = light.intensity / r_square;
    //    Eigen::Vector3f direction_point2eye = eye_pos - point;
    //    //Eigen::Vector3f light_energy2eye = light.intensity / r_square;
    //
    //    //// find the increament *ambient*
    //    L_ambient += ka.cwiseProduct(amb_light_intensity);     //Note:I_a is a constant in this case
    //
    //    // find the increament *diffuse*
    //    float diffuse_max = std::max(0.f, normal.dot(direction_point2light.normalized()));
    //    L_diffuse += kd.cwiseProduct(light_energy2point) * diffuse_max;
    //
    //    // find the increament *specular*
    //    Eigen::Vector3f h_sum_specular = direction_point2light + direction_point2eye;
    //    float magnitude_specular = h_sum_specular.norm();
    //    float specular_max = std::max(0.f, normal.dot(h_sum_specular / magnitude_specular));
    //    float specular_max_tempVector = std::pow(specular_max, p);
    //    L_specular += ks.cwiseProduct(light_energy2point) * specular_max_tempVector;
    //}
    //result_color = L_ambient + L_diffuse + L_specular;
    result_color = normal;

    return result_color * 255.f;
}


int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 145.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;                                            // mainly for loading 3D object
    //std::string obj_path = "models/spot/";
    std::string obj_path = "models/spot/";

    // Load .obj File
    // based on the file, there is all triangle information related to the normal and vertex
    bool loadout = Loader.LoadFile("models/spot/spot_triangulated_good.obj");       //3D_heart_model
    //bool loadout = Loader.LoadFile("models/spot/heart.obj");
    //bool loadout = Loader.LoadFile("models/bunny/bunny.obj");
    for(auto mesh:Loader.LoadedMeshes)          // start the classification of the triangle
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                //setVertex for the vertex of the triangle, setNormal is the normal line of the triangle, and setTexCoord is to find texture based on the position
                // v is the vertex, vn is vertex normal, and vt is vertex texture
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            // finishing creating the list for each triangle
            TriangleList.push_back(t);
        }
    }

    // start the rasterization
    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));

    // select the type of the shader
    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = normal_fragment_shader;

    if (argc >= 2)  //run the code only if the argument is satisfied, select the texture based on the input
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the displacement shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    // eye position
    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
