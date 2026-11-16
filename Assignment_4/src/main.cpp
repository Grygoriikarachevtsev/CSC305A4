////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <fstream>
#include <algorithm>
#include <numeric>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Class to store tree
////////////////////////////////////////////////////////////////////////////////
class AABBTree
{
public:
    class Node
    {
    public:
        AlignedBox3d bbox;
        int parent;   // Index of the parent node (-1 for root)
        int left;     // Index of the left child (-1 for a leaf)
        int right;    // Index of the right child (-1 for a leaf)
        int triangle; // Index of the node triangle (-1 for internal nodes)
    };

    std::vector<Node> nodes;
    int root;

    AABBTree() = default;                           // Default empty constructor
    AABBTree(const MatrixXd &V, const MatrixXi &F); // Build a BVH from an existing mesh

private:
    // builds the bvh recursively
    int build_recursive(const MatrixXd &V, const MatrixXi &F, const MatrixXd &centroids, int from, int to, int parent, std::vector<int> &triangles);
};

////////////////////////////////////////////////////////////////////////////////
// Scene setup, global variables
////////////////////////////////////////////////////////////////////////////////
const std::string data_dir = DATA_DIR;
const std::string filename("raytrace.png");
const std::string mesh_filename(data_dir + "dragon.off/dragon.off");

//Camera settings
const double focal_length = 2;
const double field_of_view = 0.7854; //45 degrees
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 2);

// Triangle Mesh
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)
AABBTree bvh;

//Material for the object, same material for all objects
const Vector4d obj_ambient_color(0.0, 0.5, 0.0, 0);
const Vector4d obj_diffuse_color(0.5, 0.5, 0.5, 0);
const Vector4d obj_specular_color(0.2, 0.2, 0.2, 0);
const double obj_specular_exponent = 256.0;
const Vector4d obj_reflection_color(0.7, 0.7, 0.7, 0);

// Precomputed (or otherwise) gradient vectors at each grid node
const int grid_size = 20;
std::vector<std::vector<Vector2d>> grid;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector4d> light_colors;

//Ambient light
const Vector4d ambient_light(0.2, 0.2, 0.2, 0);

// Bonus 1 Stuff
std::vector<Vector3d> sphere_centers;
std::vector<double> sphere_radii;
std::vector<Matrix3d> parallelograms; 

//Fills the different arrays
void setup_scene()
{
    //Loads file
    std::ifstream in(mesh_filename);
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    //setup tree
    bvh = AABBTree(vertices, facets);

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    //Spheres
    sphere_centers.emplace_back(1, 0.2, -1);
    sphere_radii.emplace_back(1);

    //parallelograms
    parallelograms.emplace_back();
    parallelograms.back() << -100, -100, 100,
        -1.25, -1.2, 0,
        -100, 100, -100;
}

////////////////////////////////////////////////////////////////////////////////
// BVH Code
////////////////////////////////////////////////////////////////////////////////

AlignedBox3d bbox_from_triangle(const Vector3d &a, const Vector3d &b, const Vector3d &c)
{
    AlignedBox3d box;
    box.extend(a);
    box.extend(b);
    box.extend(c);
    return box;
}

AABBTree::AABBTree(const MatrixXd &V, const MatrixXi &F)
{
    // Compute the centroids of all the triangles in the input mesh
    MatrixXd centroids(F.rows(), V.cols());
    centroids.setZero();
    for (int i = 0; i < F.rows(); ++i)
    {
        for (int k = 0; k < F.cols(); ++k)
        {
            centroids.row(i) += V.row(F(i, k));
        }
        centroids.row(i) /= F.cols();
    }

    //Vector containing the list of tringle indices
    std::vector<int> triangles(F.rows());
    std::iota(triangles.begin(), triangles.end(), 0);

    root = build_recursive(V, F, centroids, 0, triangles.size(), -1, triangles);
}

int AABBTree::build_recursive(const MatrixXd &V, const MatrixXi &F, const MatrixXd &centroids, int from, int to, int parent, std::vector<int> &triangles)
{
    // Scene is empty, so is the aabb tree
    if (to - from == 0)
    {
        return -1;
    }

    // If there is only 1 triangle left, then we are at a leaf
    if (to - from == 1)
    {
        //TODO create leaf node and retun correct left index
        const int triangle_id = triangles[from];

        const Vector3d a = V.row(F(triangle_id, 0)).transpose();
        const Vector3d b = V.row(F(triangle_id, 1)).transpose();
        const Vector3d c = V.row(F(triangle_id, 2)).transpose();

        Node leaf; 
        leaf.bbox = bbox_from_triangle(a, b, c);
        leaf.parent = parent;
        leaf.left = -1;
        leaf.right = -1;
        leaf.triangle = triangles[from];

        const int node_index = static_cast<int>(nodes.size());
        nodes.push_back(leaf);

        return node_index;
    }

    // TODO sort centroids along the longest dimension

    //TODO Use AlignedBox3d to find the box around the current centroids
    AlignedBox3d centroid_box;
    for(int i = from; i < to; ++i){
        const int triangle_id = triangles[i];
        const Vector3d center = centroids.row(triangle_id).transpose();

        centroid_box.extend(center);
    }

    // Diagonal of the box
    Vector3d extent = centroid_box.diagonal();

    //TODO find the largest dimension
    int longest_dim = 0;
    for(int dim = 1; dim < 3; ++dim){
        if(extent(dim) > extent(longest_dim)) longest_dim = dim;
    }

    std::sort(triangles.begin() + from, triangles.begin() + to, [&](int f1, int f2) {
        //TODO sort the **triangles** along the centroid largest dimension
        // return true if triangle f1 comes before triangle f2
        return centroids(f1, longest_dim) < centroids(f2, longest_dim);
    });

    //TODO Create a new internal node and do a recursive call to build the left and right part of the tree
    Node internal;
    internal.parent = parent;
    internal.left = -1;
    internal.right = -1;
    internal.triangle = -1;

    const int node_index = static_cast<int>(nodes.size());
    nodes.push_back(internal);

    const int mid = from + (to - from + 1) / 2; // divides the range

    const int left_index = build_recursive(V, F, centroids, from, mid, node_index, triangles);
    const int right_index = build_recursive(V, F, centroids, mid, to, node_index, triangles);

    nodes[node_index].left = left_index;
    nodes[node_index].right = right_index;

    nodes[node_index].bbox = nodes[left_index].bbox;
    nodes[node_index].bbox.extend(nodes[right_index].bbox);

    //TODO finally return the correct index
    return node_index;
}

////////////////////////////////////////////////////////////////////////////////
// Intersection code
////////////////////////////////////////////////////////////////////////////////

double ray_triangle_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, const Vector3d &a, const Vector3d &b, const Vector3d &c, Vector3d &p, Vector3d &N)
{
    // TODO
    // Compute whether the ray intersects the given triangle.
    // If you have done the parallelogram case, this should be very similar to it.
    const Vector3d e1 = b - a;
    const Vector3d e2 = c - a;
    
    Matrix3d system;
    
    system.col(0) = e1;
    system.col(1) = e2;
    system.col(2) = -ray_direction;
    
    const double eps = 1e-8;
    
    if(std::abs(system.determinant()) < eps) return -1;
    
    const Vector3d rhs = ray_origin - a;
    const Vector3d sol = system.colPivHouseholderQr().solve(rhs);

    const double u = sol(0);
    const double v = sol(1);
    const double t = sol(2);

    if(t < 0.0) return -1;
    if(u < 0.0 || v < 0.0) return -1;
    if(u + v > 1.0) return -1;

    p = ray_origin + t * ray_direction;

    N = e1.cross(e2).normalized(); // necessary for shadows to work properly
    // N = e2.cross(e1).normalized(); // outside facing side have negative dot product

    return t;
}

bool ray_box_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, const AlignedBox3d &box)
{
    // TODO
    // Compute whether the ray intersects the given box.
    // we are not testing with the real surface here anyway.
    double t_enter = 0;
    double t_exit = std::numeric_limits<double>::max();
    for(std::size_t i = 0; i < 3; ++i){
        if(ray_direction(i) == 0){
            if(box.min()(i) > ray_origin(i) || box.max()(i) < ray_origin(i)) return false;
            continue;
        }

        const double t0 = (box.min()(i) - ray_origin(i)) / ray_direction(i);
        const double t1 = (box.max()(i) - ray_origin(i)) / ray_direction(i);

        const double axis_enter = std::min(t0, t1);
        const double axis_exit = std::max(t0, t1);

        t_enter = std::max(t_enter, axis_enter);
        t_exit = std::min(t_exit, axis_exit);

        if(t_enter > t_exit) return false;
    }

    return true;
}

//Compute the intersection between a ray and a sphere, return -1 if no intersection
double ray_sphere_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, int index, Vector3d &p, Vector3d &N)
{
    // TODO, implement the intersection between the ray and the sphere at index index.
    //return t or -1 if no intersection

    const Vector3d sphere_center = sphere_centers[index];
    const Vector3d center_to_ray_origin = ray_origin - sphere_center;
    
    double t = -1;
    const double sphere_radius = sphere_radii[index];

    const double a = ray_direction.dot(ray_direction);
    const double b = 2 * ray_direction.dot(center_to_ray_origin);
    const double c = center_to_ray_origin.dot(center_to_ray_origin) - sphere_radius * sphere_radius;

    const double discriminant = b * b - 4.0 * a * c;

    if(discriminant >= 0){
        const double square_root = std::sqrt(discriminant);
        
        const double t1 = (-b - square_root) / (2.0 * a);
        const double t2 = (-b + square_root) / (2.0 * a);

        if(t1 >= 0.0) t = t1;
        else if(t2 >= 0.0) t = t2;

        if(t >= 0.0){
            p = ray_origin + t * ray_direction; // ray intersection
            N = (p - sphere_center).normalized(); // ray normal
        }
    }

    return t;
}

//Compute the intersection between a ray and a paralleogram, return -1 if no intersection
double ray_parallelogram_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, int index, Vector3d &p, Vector3d &N)
{
    // TODO, implement the intersection between the ray and the parallelogram at index index.
    //return t or -1 if no intersection

    const Vector3d pgram_origin = parallelograms[index].col(0);
    const Vector3d A = parallelograms[index].col(1);
    const Vector3d B = parallelograms[index].col(2);
    const Vector3d pgram_u = A - pgram_origin;
    const Vector3d pgram_v = B - pgram_origin;


    Matrix3d system;

    system.col(0) = ray_direction;
    system.col(1) = -pgram_u;
    system.col(2) = -pgram_v;

    const double eps = 1e-8;

    if(std::abs(system.determinant()) < eps) return -1; // ray is parallel

    const Vector3d rhs = pgram_origin - ray_origin;
    const Vector3d sol = system.colPivHouseholderQr().solve(rhs);

    const double t = sol(0);
    const double alpha = sol(1);
    const double beta = sol(2);

    if(t < 0.0) return -1;

    if(alpha < 0.0 || alpha > 1.0) return -1;
    if(beta < 0.0 || beta > 1.0) return -1;


    //TODO set the correct intersection point, update p and N to the correct values
    p = ray_origin + t * ray_direction;
    N = pgram_u.cross(pgram_v).normalized();
    
    return t;
}


//Finds the closest intersecting object returns its index
//In case of intersection it writes into p and N (intersection point and normals)
bool find_nearest_object(const Vector3d &ray_origin, const Vector3d &ray_direction, Vector3d &p, Vector3d &N)
{
    Vector3d tmp_p, tmp_N;
    double closest_t = std::numeric_limits<double>::max();

    // TODO
    // Method (1): Traverse every triangle and return the closest hit.

    // for(std::size_t i = 0; i < facets.rows(); ++i){
    //     const Vector3d a = vertices.row(facets(i, 0)).transpose();
    //     const Vector3d b = vertices.row(facets(i, 1)).transpose();
    //     const Vector3d c = vertices.row(facets(i, 2)).transpose();

    //     const double t = ray_triangle_intersection(ray_origin, ray_direction, a, b, c, tmp_p, tmp_N);
        
    //     if(t >= 0){
    //         if(t < closest_t) {
    //             closest_t = t;
    //             p = tmp_p;
    //             N = tmp_N;
    //         }
    //     }
    // }

    // Method (2): Traverse the BVH tree and test the intersection with a
    // triangles at the leaf nodes that intersects the input ray.
    
    // if(bvh.root == -1) return false;

    std::vector<int> pending;

    // An empty mesh does not mean an empty scene now
    if(bvh.root != -1) {
        pending.push_back(bvh.root);    // encloses the mesh, initially it is the only node waiting for examination
    }

    while(!pending.empty()){
        // take node to examine it
        const int node_index = pending.back();
        pending.pop_back();

        // node access
        const AABBTree::Node &node = bvh.nodes[node_index];

        // if the node misses the box, it cannot hit any enclosed triangle, move onto the next pending node
        if(!ray_box_intersection(ray_origin, ray_direction, node.bbox)) continue;

        // check for a leaf, triangle 0 is valid
        if(node.triangle >= 0){
            const int triangle_id = node.triangle;

            const Vector3d a = vertices.row(facets(triangle_id, 0)).transpose();
            const Vector3d b = vertices.row(facets(triangle_id, 1)).transpose();
            const Vector3d c = vertices.row(facets(triangle_id, 2)).transpose();

            const double t = ray_triangle_intersection(ray_origin, ray_direction, a, b, c, tmp_p, tmp_N);

            if(t >= 0 && t < closest_t){
                closest_t = t;
                p = tmp_p;
                N = tmp_N;
            }
        } else {
            pending.push_back(node.right);
            pending.push_back(node.left);
        }

    }
    
    // sphere loop
    for(int i = 0; i < static_cast<int>(sphere_centers.size()); ++i){
        const double t = ray_sphere_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);

        if(t >= 0.0 && t < closest_t){
            closest_t = t;
            p = tmp_p;
            N = tmp_N;
        }
    }

    // parallelogram loop
    for(int i = 0; i < static_cast<int>(parallelograms.size()); ++i){
        const double t = ray_parallelogram_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);

        if(t >= 0.0 && t < closest_t){
            closest_t = t;
            p = tmp_p;
            N = tmp_N;
        }
    }

    return closest_t != std::numeric_limits<double>::max();
}

////////////////////////////////////////////////////////////////////////////////
// Raytracer code
////////////////////////////////////////////////////////////////////////////////

bool is_light_visible(const Vector3d &ray_origin, const Vector3d &ray_direction, const Vector3d &light_position)
{
    // TODO: Determine if the light is visible here
    // Use find_nearest_object
    Vector3d hit_p, hit_N;
    const bool hit = find_nearest_object(ray_origin, ray_direction, hit_p, hit_N);
    if(!hit) return true;

    const double hit_distance = (hit_p - ray_origin).norm();
    const double light_distance = (light_position - ray_origin).norm();

    return hit_distance >= light_distance;
}


Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction, int max_bounce)
{
    //Intersection point and normal, these are output of find_nearest_object
    Vector3d p, N;

    const bool nearest_object = find_nearest_object(ray_origin, ray_direction, p, N);

    if (!nearest_object)
    {
        // Return a transparent color
        return Vector4d(0, 0, 0, 0);
    }

    // Ambient light contribution
    const Vector4d ambient_color = obj_ambient_color.array() * ambient_light.array();

    // Punctual lights contribution (direct lighting)
    Vector4d lights_color(0, 0, 0, 0);
    for (int i = 0; i < light_positions.size(); ++i)
    {
        const Vector3d &light_position = light_positions[i];
        const Vector4d &light_color = light_colors[i];

        Vector4d diff_color = obj_diffuse_color;

        // Diffuse contribution
        const Vector3d Li = (light_position - p).normalized();
        const Vector4d diffuse = diff_color * std::max(Li.dot(N), 0.0);

        const double shadow_bias = 1e-4; 
        const double normal_side = Li.dot(N) >= 0.0 ? 1.0 : -1.0; // alignment with the normal
        const Vector3d shadow_origin = p + normal_side * shadow_bias * N;
        const Vector3d shadow_direction = (light_position - shadow_origin).normalized();
        if(!is_light_visible(shadow_origin, shadow_direction, light_position)) continue;


        // Specular contribution
        const Vector3d Hi = (Li - ray_direction).normalized();
        const Vector4d specular = obj_specular_color * std::pow(std::max(N.dot(Hi), 0.0), obj_specular_exponent);
        // Vector3d specular(0, 0, 0);

        // Attenuate lights according to the squared distance to the lights
        const Vector3d D = light_position - p;
        lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
    }

    Vector4d reflection_color = Vector4d::Zero();

    if(max_bounce > 0) {

        Vector3d reflected_direction = (ray_direction - 2.0 * ray_direction.dot(N) * N).normalized();

        const double bias = 1e-4;
        double side = reflected_direction.dot(N) >= 0.0 ? 1.0 : -1.0;
        Vector3d reflected_origin = p + side * bias * N;

        Vector4d seen_color = shoot_ray(reflected_origin, reflected_direction, max_bounce - 1);
        reflection_color = obj_reflection_color.cwiseProduct(seen_color);
    }

    // Rendering equation
    Vector4d C = ambient_color + lights_color + reflection_color;

    //Set alpha to 1
    C(3) = 1;

    return C;
}

////////////////////////////////////////////////////////////////////////////////

void raytrace_scene()
{
    std::cout << "Simple ray tracer." << std::endl;

    int w = 640;
    int h = 480;
    MatrixXd R = MatrixXd::Zero(w, h);
    MatrixXd G = MatrixXd::Zero(w, h);
    MatrixXd B = MatrixXd::Zero(w, h);
    MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask

    // The camera always points in the direction -z
    // The sensor grid is at a distance 'focal_length' from the camera center,
    // and covers an viewing angle given by 'field_of_view'.
    double aspect_ratio = double(w) / double(h);
    //TODO
    double image_y = std::tan(field_of_view / 2.0) * focal_length;
    double image_x = image_y * aspect_ratio;

    // The pixel grid through which we shoot rays is at a distance 'focal_length'
    const Vector3d image_origin(-image_x, image_y, camera_position[2] - focal_length);
    const Vector3d x_displacement(2.0 / w * image_x, 0, 0);
    const Vector3d y_displacement(0, -2.0 / h * image_y, 0);

    for (unsigned i = 0; i < w; ++i)
    {
        for (unsigned j = 0; j < h; ++j)
        {
            const Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;

            // Prepare the ray
            Vector3d ray_origin;
            Vector3d ray_direction;

            if (is_perspective)
            {
                // Perspective camera
                ray_origin = camera_position;
                ray_direction = (pixel_center - camera_position).normalized();
            }
            else
            {
                // Orthographic camera
                ray_origin = pixel_center;
                ray_direction = Vector3d(0, 0, -1);
            }

            const Vector4d C = shoot_ray(ray_origin, ray_direction, 5);
            R(i, j) = C(0);
            G(i, j) = C(1);
            B(i, j) = C(2);
            A(i, j) = C(3);
        }
    }

    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    setup_scene();

    raytrace_scene();
    return 0;
}
