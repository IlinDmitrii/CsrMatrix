#include "CsrMatrix.hpp"
#include <random>
#include <chrono>
#include <format>
int main(){
    //создание через initializer_list
    CsrMatrix<double> A{
    {4, 1, 0},
    {1, 3, 0},
    {0, 0, 2}
    };
    std::cout << "Матрица A:\n" << A << "\n";

    //проверка сложения
    CsrMatrix<double> E{
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1}
    };
    auto C = A + E;
    std::cout << "A + E:\n" << C << "\n";

    //проверка умножения на вектор
    using vector_type=typename CsrMatrix<double>::vector_type;
    using index_type=typename CsrMatrix<double>::index_type;
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


    size_t n = 100000;

    CsrMatrix<double> R(n,n);
    // Тест времени создания больших матриц
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        R.set(i,i,4.0);
        if (i>0) R.set(i,i-1,-1.0);
        if (i<n-1) R.set(i,i+1,-1.0);
        if (i<n-2) R.set(i,i+2,-0.5);
        if (i < n-2) R.set(i, i+2, -0.5);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::format("\nВремя построения матрицы {} * {} через set-ы: \n{} секунд\n\n",
    n,n,std::chrono::duration<double>(end - start).count());

    R={};

    start = std::chrono::high_resolution_clock::now();
    vector_type values;
    std::vector<index_type> col_indices;
    std::vector<index_type> row_ptr(n+1, 0);

    for (size_t i = 0; i < n; ++i) {
        row_ptr[i] = values.size();
        if (i > 0) {
            values.push_back(-1.0);
            col_indices.push_back(i-1);
        }
        values.push_back(4.0);
        col_indices.push_back(i);
        if (i < n-1) {
            values.push_back(-1.0);
            col_indices.push_back(i+1);
        }
        if (i < n-2) {
            values.push_back(-0.5);
            col_indices.push_back(i+2);
        }
    }
    row_ptr[n] = values.size();

    R=CsrMatrix<double>(n, n, values, col_indices, row_ptr);

    end = std::chrono::high_resolution_clock::now();
    std::cout << std::format("Время построения матрицы {} * {} через явное заполнение векторов: \n{} секунд\n\n",
    n,n,std::chrono::duration<double>(end - start).count());

    vector_type rhs(n, 1.0);
    // Тест решения больших матриц
    start = std::chrono::high_resolution_clock::now();
    x = BiCGStab(R, rhs, {}, 1e-8, 100000);
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << std::format(
    "Время решения BiCGStab для четырехдиагональной матрицы {}*{} с примерно {} элементами:\n{} секунд\n",
    n, n, 4*n, elapsed.count());

    return 0;
}
