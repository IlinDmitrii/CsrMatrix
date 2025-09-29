#pragma once

#include <vector>
#include <cstddef>
#include <initializer_list>
#include <string>
#include <iostream>

template<typename T = double>
class CsrMatrix {
public:

    using value_type      = T;
    using index_type      = std::size_t;
    using vector_type = std::vector<value_type>;

    CsrMatrix() noexcept = default;
    CsrMatrix(index_type rows, index_type cols);
    CsrMatrix(index_type rows, index_type cols,
              std::vector<value_type> values,
              std::vector<index_type>  col_indices,
              std::vector<index_type>  row_ptr);

    explicit CsrMatrix(const std::vector<std::vector<value_type>>& dense);
    CsrMatrix(std::initializer_list<std::initializer_list<value_type>> init);


    CsrMatrix(const CsrMatrix& other);
    CsrMatrix(CsrMatrix&& other) noexcept;
    CsrMatrix& operator=(const CsrMatrix& other);
    CsrMatrix& operator=(CsrMatrix&& other) noexcept;
    ~CsrMatrix() = default;


    value_type at(index_type row, index_type col) const;
    void       set(index_type row, index_type col, value_type v);
    vector_type row(index_type r) const;


    index_type rows() const noexcept { return rows_; }
    index_type cols() const noexcept { return cols_; }
    index_type nonzeros() const noexcept { return values_.size(); }


    [[nodiscard]]
    CsrMatrix operator+(const CsrMatrix& rhs) const;
    [[nodiscard]]
    CsrMatrix operator-(const CsrMatrix& rhs) const;

    template <typename Scalar>
    [[nodiscard]]
    CsrMatrix operator*(Scalar scalar) const;



    std::vector<value_type> multiply(const std::vector<value_type>& vec) const;
    CsrMatrix   multiply(const CsrMatrix& rhs) const;


    [[nodiscard]]
    CsrMatrix transpose() const;


private:
    index_type rows_ {0};
    index_type cols_ {0};
    std::vector<value_type> values_;      // Ненулевые элементы
    std::vector<index_type> col_indices_; // Столбцы для каждого значения
    std::vector<index_type> row_ptr_;     // Индексы начала строк в values_
};

#include "CsrMatrix.inl"
