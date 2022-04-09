/* Copyright (C) 2020-2022  Dhruv Chawla */
/* See LICENSE at project root for license details */
#include <iostream>
#include <matty.hpp>

using namespace matty;

template <typename T>
void print(const DynamicMatrix<T> &matrix) {
    for (std::size_t i = 0; i < matrix.rows(); i++) {
        for (std::size_t j = 0; j < matrix.columns(); j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << '\n';
    }
}

template <typename T>
void iota(DynamicMatrix<T> &matrix) {
    for (std::size_t i = 0; i < matrix.rows(); i++) {
        for (std::size_t j = 0; j < matrix.columns(); j++) {
            matrix[i][j] = i * (matrix.columns() - 1) + j;
        }
    }
}

template <typename T>
void iota_reverse(DynamicMatrix<T> &matrix) {
    for (std::size_t i = 0; i < matrix.rows(); i++) {
        for (std::size_t j = 0; j < matrix.columns(); j++) {
            matrix[i][j] = (matrix.rows() - i - 1) * (matrix.columns() - 1) + (matrix.columns() - j - 1);
        }
    }
}

template <typename T>
void print(const FixedMatrix4<T> &matrix) {
    for (std::size_t i = 0; i < matrix.rows(); i++) {
        for (std::size_t j = 0; j < matrix.columns(); j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << '\n';
    }
}

using MatrixType = DynamicMatrix<int>;

int main() {
    DynamicMatrix m1{11, 7};
    iota(m1);

    DynamicMatrix m2{11, 7};
    iota_reverse(m2);

    std::cout << "\nm1:\n--------\n";
    print(m1);

    std::cout << "\nm2:\n--------\n";
    print(m2);

    DynamicMatrix sum = m1.add(m2);
    std::cout << "\nsum:\n--------\n";
    print(sum);

    DynamicMatrix m3{{2, 7}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    std::cout << "\nm3:--------\n";
    print(m3);

    DynamicMatrix m4{{7, 3}, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    std::cout << "\nm4:--------\n";
    print(m4);

    std::cout << "\nm5:\n--------\n";
    DynamicMatrix m5 = m3.mul(m4);
    print(m5);

    DynamicMatrix m6{{3, 3}, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::cout << "\nm6:--------\n";
    print(m6);

    std::cout << "\nm7:\n--------\n";
    DynamicMatrix m7 = m6.transpose();
    print(m7);

    std::cout << "\nSIMD status: " << MATTY_SIMD_STATUS << '\n';

    DynamicMatrix m8{{3, 3}, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::cout << "\nm8:--------\n";
    m8.horizontal_flip_self();
    print(m8);

    DynamicMatrix m9{{3, 3}, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::cout << "\nm9:--------\n";
    m9.vertical_flip_self();
    print(m9);

    FixedMatrix4<int> x1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    //    FixedMatrix4<int> x2 = FixedMatrix4<int>::identity().add(FixedMatrix4<int>::identity());
    FixedMatrix4<int> x2 = FixedMatrix4<int>::identity();
    FixedMatrix4<int> x3 = x1.mul(x2);

    std::cout << "\nx1:--------\n";
    print(x1);
    std::cout << "\nx2:--------\n";
    print(x2);
    std::cout << "\nx3:--------\n";
    print(x3);

    FixedMatrix4<int> x5 = FixedMatrix4<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    x5.transpose_self();
    std::cout << "\nx5:--------\n";
    print(x5);

    FixedMatrix4<float> x6 = FixedMatrix4<float>{
        1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f, 12.5f, 13.5f, 14.5f, 15.5f, 16.5f};
    FixedMatrix4<float> x7 = FixedMatrix4<float>::identity().mul_self(x6).add_self(1.0f).mul_self(3.0f);
    FixedMatrix4<float> x8 = x6.mul(x7);

    std::cout << "\nx6:--------\n";
    print(x6);
    std::cout << "\nx7:--------\n";
    print(x7);
    std::cout << "\nx8:--------\n";
    print(x8);

    return 0;
}