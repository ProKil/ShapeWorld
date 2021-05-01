#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <cmath>
#include <iostream>

namespace py = pybind11;
typedef std::pair<double, double> Point;
typedef std::tuple<double, double, double> Color;

double golden_ratio = std::sqrt(5.0) / 2.0 - 0.5;

Point operator+(Point x, double y) {
    return Point(x.first + y, x.second + y);
}

Point operator+(Point x, Point y) {
    return Point(x.first + y.first, x.second + y.second);
}

Point operator-(Point x, double y) {
    return Point(x.first - y, x.second - y);
}

Point operator-(Point x, Point y) {
    return Point(x.first - y.first, x.second - y.second);
}

Point operator*(Point x, double y) {
    return Point(x.first * y, x.second * y);
}

Point operator*(Point x, Point y) {
    return Point(x.first * y.first, x.second * y.second);
}

Point operator/(Point x, Point y) {
    return Point(x.first / y.first, x.second / y.second);
}

Point operator/(Point x, double y) {
    return Point(x.first / y, x.second / y);
}

Point pos(Point x) {
    return Point(x.first>0?x.first:0, x.second>0?x.second:0);
}

Point cap(Point x, Point y) {
    return Point(x.first<y.first?x.first:y.first,
                 x.second<y.second?x.second:y.second);
}

Point abs(Point x) {
    return Point(std::abs(x.first), std::abs(x.second));
}

double len(Point x) {
    return std::sqrt(std::pow(x.first, 2) + std::pow(x.second, 2));
}

Point rotate(Point x, Point rotation) {
    return Point(x.first * rotation.second
                 - x.second * rotation.first,
                 x.first * rotation.first
                 + x.second * rotation.second);
}

double calc_distance(const Point& offset, const std::string& shape, const Point& size) {
    if (shape=="square" || shape=="rectangle") {
        return len(pos(abs(offset) - size));
    } else if (shape=="triangle") {
        if (offset.second < -size.second) return len(pos(abs(offset) - size));
        else {
            auto new_offset = Point(std::abs(offset.first), offset.second + size.second);
            auto linear = std::min(std::max(new_offset.second - new_offset.first + size.first, 0.0) / (size.first + 2.0 * size.second), 1.0);
            auto new_size = Point((1.0 - linear) * size.first, linear * 2.0 * size.second);
            return len(pos(new_offset - new_size));
        }
    } else if (shape=="pentagon") {
        auto new_offset = Point(std::abs(offset.first),
                                offset.second + size.second 
                                - golden_ratio * 2.0 * size.second);
        if (new_offset.second < 0.0) {
            auto y_length = golden_ratio * 2.0 * size.second;
            if (new_offset.first < golden_ratio * size.first) {
                return std::max(-new_offset.second - y_length, 0.0);
            } else {
                new_offset = Point(
                    new_offset.first - golden_ratio * size.first,
                    -new_offset.second);
                auto x_length = (1.0 - golden_ratio) * size.first;
                auto linear = std::min(
                    std::max(
                        new_offset.second - new_offset.first + x_length, 0.0)
                        / (x_length + y_length), 
                    1.0
                );
                auto new_size = Point(
                    (1.0 - linear) * x_length, linear * y_length);
                return len(pos((new_offset - new_size)));
            }
        } else {
            auto y_length = (1.0 - golden_ratio) * 2.0 * size.second;
            auto linear = std::min(
                std::max(
                    new_offset.second - new_offset.first + size.first,
                    0.0
                ) / (size.first + y_length), 
                1.0
            );
            auto new_size = Point(
                (1.0 - linear) * size.first, linear * y_length);
            return len(pos(new_offset - new_size));
        } 
    } else if (shape=="cross") {
        auto new_offset = abs(offset);
        if (new_offset.first > new_offset.second) {
            auto new_size = Point(size.first, size.second / 3);
            return len(pos(new_offset - new_size));
        } else {
            auto new_size = Point(size.first / 3, size.second);
            return len(pos(new_offset - new_size));
        }
    } else if (shape=="circle") {
        return std::max(len(offset) - size.first, 0.0);
    } else if (shape == "ellipse") {
        auto new_offset = abs(offset) / size;
        new_offset = new_offset - new_offset / len(new_offset);
        return len(pos(new_offset * size));
    } else if (shape == "semicircle") {
        auto new_offset = offset + Point(0.0, size.second);
        if (new_offset.second < 0.0)
            return len(pos(abs(new_offset) - Point(size.first, 0.0)));
        else
            return std::max(len(new_offset) - size.first, 0.0);
    }else {
        throw std::runtime_error(shape);
    }
}



void draw(
    py::array_t<float> world_array,
    Point world_size,
    Point topleft,
    Point bottomright,
    Color color,
    Point center,
    std::string shape,
    Point rotation,
    Point size
    ) {
    
    // check array 
    auto r = world_array.mutable_unchecked<3>(); 

    // constants
    auto shift = Point(2.0, 2.0) / world_size;
    auto scale = shift * 2.0 + 1.0;
    auto topleft_ = pos((topleft / scale) * world_size);
    auto bottomright_ = cap(
        ((bottomright + shift * 2.0) / scale) * world_size,
        world_size
    );

    double world_length = std::min(world_size.first, world_size.second);

    for (size_t y = size_t(topleft_.second); y < size_t(bottomright_.second); y ++) 
        for (size_t x = size_t(topleft_.first); x < size_t(bottomright_.first); x ++) {
            auto point = Point(x, y) / (world_size - 1.0);
            point = point * scale - shift;
            auto offset = point - center;
            auto rotated_offset = rotate(offset, rotation);
            auto distance = calc_distance(rotated_offset , shape, size);
            if (distance == 0.0) {
                r(y, x, 0) = std::get<0>(color);
                r(y, x, 1) = std::get<1>(color);
                r(y, x, 2) = std::get<2>(color);
            } else {
                distance = std::max(1.0 - distance * world_length, 0.0);
                r(y, x, 0) = distance * std::get<0>(color) + (1.0 - distance) * r(y, x, 0);
                r(y, x, 1) = distance * std::get<1>(color) + (1.0 - distance) * r(y, x, 1);
                r(y, x, 2) = distance * std::get<2>(color) + (1.0 - distance) * r(y, x, 2);
            }
        }
    return;
}

PYBIND11_MODULE(drawcpp, m) {
    m.def("draw", &draw);
}