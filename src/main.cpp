#include "CsrMatrix.hpp"

int main(){
    //создание через initializer_list
    CsrMatrix<double> A{
    {4, 1, 0},
    {1, 3, 0},
    {0, 0, 2}
    };
    std::cout << "Матрица A:\n" << A << "\n";

    //проверка сложения и вычитания
    CsrMatrix<double> E{
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1}
    };
    auto C = A + E;
    auto D = A - E;
    std::cout << "A + E:\n" << C << "\n";
    std::cout << "A - E:\n" << D << "\n";

    //проверка умножения на вектор
    using vector_type=typename CsrMatrix<double>::vector_type;
    vector_type vec{1, 2, 3};
    auto result = A.multiply(vec);
    std::cout << "A * [1 2 3]^T = ";
    std::cout<<"[ ";
    for (auto v : result) std::cout << v << " ";
    std::cout << "]^T\n";

    //проверка умножения матрицы на матрицу
    std::cout<< "\nA*(A + E):\n";
    std::cout<<A.multiply(C)<<"\n";


    //проверка решателя СЛАУ
    vector_type b{5, 6, 4};
    auto x = BiCGStab(A, b);

    std::cout << "Решение Ax = b:\n b = [ ";
    for (auto v : b) std::cout << v << " ";
    std::cout << "]^T\n x = [ ";
    for (auto v : x) std::cout << v << " ";
    std::cout << "]^T\n";

    // Проверка корректности решения: A*x ≈ b
    auto check = A.multiply(x);
    std::cout << "Проверка A*x = [ ";
    for (auto v : check) std::cout << v << " ";
    std::cout << "]^T\n";

    return 0;
}
