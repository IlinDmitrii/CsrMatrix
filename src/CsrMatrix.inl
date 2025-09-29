#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <initializer_list>
#include <utility>
#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <limits>

template<typename T>
inline CsrMatrix<T>::CsrMatrix(index_type rows, index_type cols)
    :   rows_(rows),
        cols_(cols)
{
    row_ptr_.resize(rows_ + 1, 0);
}

template<typename T>
inline CsrMatrix<T>::CsrMatrix(index_type rows, index_type cols,
                               std::vector<value_type> values,
                               std::vector<index_type> col_indices,
                               std::vector<index_type> row_ptr)
    : rows_(rows), cols_(cols),
      values_(std::move(values)),
      col_indices_(std::move(col_indices)),
      row_ptr_(std::move(row_ptr))
{
    if (row_ptr_.size() != rows_ + 1) {
        throw std::invalid_argument("row_ptr != rows + 1");
    }
    if (values_.size() != col_indices_.size()) {
        throw std::invalid_argument("values.size() !=  col_indices.size()");
    }
}


template<typename T>
inline CsrMatrix<T>::CsrMatrix(const std::vector<std::vector<value_type>>& dense)
    :   rows_(dense.size()),
        cols_(dense.empty() ? 0 : dense[0].size())
{
    row_ptr_.resize(rows_ + 1, 0);

    for (index_type r = 0; r < rows_; ++r) {
        if (dense[r].size() != cols_) {
            throw std::invalid_argument("invalid number of cols in a row");
        }

        for (index_type c = 0; c < cols_; ++c) {
            const value_type& val = dense[r][c];
            if (val != value_type{}) {
                values_.push_back(val);
                col_indices_.push_back(c);
            }
        }
        row_ptr_[r + 1] = values_.size();
    }
}


template<typename T>
inline CsrMatrix<T>::CsrMatrix(std::initializer_list<std::initializer_list<value_type>> init)
    : rows_(init.size()),
      cols_(init.size() > 0 ? init.begin()->size() : 0)
{
    row_ptr_.resize(rows_ + 1, 0);

    index_type r = 0;
    for (const auto& row : init) {
        if (row.size() != cols_) {
            throw std::invalid_argument("invalid number of cols in a row");
        }

        for (index_type c = 0; c < cols_; ++c) {
            const value_type& val = *(std::next(row.begin(), c));
            if (val != value_type{}) {
                values_.push_back(val);
                col_indices_.push_back(c);
            }
        }

        row_ptr_[r + 1] = values_.size();
        ++r;
    }
}

template<typename T>
inline CsrMatrix<T>::CsrMatrix(const CsrMatrix& other)
    :   rows_(other.rows_),
        cols_(other.cols_),
        values_(other.values_),
        col_indices_(other.col_indices_),
        row_ptr_(other.row_ptr_){
}

template<typename T>
inline CsrMatrix<T>::CsrMatrix(CsrMatrix&& other) noexcept
    :   rows_(other.rows_),
        cols_(other.cols_),
        values_(std::move(other.values_)),
        col_indices_(std::move(other.col_indices_)),
        row_ptr_(std::move(other.row_ptr_)){
    other.rows_=0;
    other.cols_=0;
}


template<typename T>
inline CsrMatrix<T>& CsrMatrix<T>::operator=(const CsrMatrix& other)
{
    if (this!=&other){
        rows_=other.rows_;
        cols_=other.cols_;
        values_=other.values_;
        col_indices_=other.col_indices_;
        row_ptr_=other.row_ptr_;
    }
    return *this;
}

template<typename T>
inline CsrMatrix<T>& CsrMatrix<T>::operator=(CsrMatrix&& other) noexcept
{
    if (this!=&other){
    rows_=other.rows_;
    other.rows_=0;

    cols_=other.cols_;
    other.cols_=0;

    values_=std::move(other.values_);
    col_indices_=std::move(other.col_indices_);
    row_ptr_=std::move(other.row_ptr_);
    }
    return *this;
}

template<typename T>
inline typename CsrMatrix<T>::value_type CsrMatrix<T>::at(index_type row, index_type col) const
{
    if (row>rows_ || col >cols_) throw std::out_of_range("indices out of matrixs indices ");

    index_type row_start = row_ptr_[row];
    index_type row_end   = row_ptr_[row + 1];

    for (index_type i=row_start;i<row_end;++i){
        if (col_indices_[i]==col) return values_[i];
    }
    return value_type{};
}

template<typename T>
inline void CsrMatrix<T>::set(  index_type row,
                                index_type col,
                                value_type v)
{
    if (row>rows_ || col >cols_) throw std::out_of_range("indices out of matrixs indices ");

    index_type row_start = row_ptr_[row];
    index_type row_end   = row_ptr_[row + 1];

    for (index_type i=row_start;i<row_end;++i){
        if (col_indices_[i]==col) {
            values_[i]=v;
            return;
        }
    }

    index_type insert_pos = row_start;
    while (insert_pos <row_end && col_indices_[insert_pos]<col)    {   ++insert_pos;   }

    values_.insert(values_.begin() + insert_pos, v);
    col_indices_.insert(col_indices_.begin() + insert_pos, col);

    for (index_type r = row + 1; r <= rows_; ++r) {
        ++row_ptr_[r];
    }
}

template<typename T>
inline typename CsrMatrix<T>::vector_type CsrMatrix<T>::row(index_type r) const
{
    if(r>=rows_) throw std::out_of_range("row number is greater than rows_");
    index_type row_start=row_ptr_[r];
    index_type row_end=row_ptr_[r+1];
    vector_type answ(cols_,value_type{});
    for (index_type i=row_start;i<row_end;++i){
        answ[col_indices_[i]]=values_[i];
    }
    return answ;
}


template<typename T>
inline CsrMatrix<T> CsrMatrix<T>::operator+(const CsrMatrix& rhs) const
{
    if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
        throw std::invalid_argument("dimentions do not match");
    }

    CsrMatrix<T> result(rows_,cols_);

    std::vector<value_type> new_values;
    std::vector<index_type> new_col_indices;

    for (index_type r = 0; r < rows_; ++r) {
        std::unordered_map<index_type, value_type> row_map; //все элементы одной новой строки вида  (col, value)
        index_type row_start_lhs=row_ptr_[r];
        index_type row_end_lhs=row_ptr_[r+1];

        for (index_type i = row_start_lhs; i < row_end_lhs; ++i) {
            row_map[col_indices_[i]] = values_[i];
        }
        index_type row_start_rhs=rhs.row_ptr_[r];
        index_type row_end_rhs=rhs.row_ptr_[r+1];

        for (index_type i = row_start_rhs; i < row_end_rhs; ++i) {
            row_map[rhs.col_indices_[i]] += rhs.values_[i];
        }

        std::vector<std::pair<index_type, value_type>> sorted_row(row_map.begin(), row_map.end());
        std::sort(sorted_row.begin(), sorted_row.end(), [](auto& a, auto& b){ return a.first < b.first; });

        for (auto& elem : sorted_row) {
            if (elem.second != value_type{}) {
                new_col_indices.push_back(elem.first);
                new_values.push_back(elem.second);
            }
        }

        result.row_ptr_[r + 1] = new_values.size();
    }

    result.values_ = std::move(new_values);
    result.col_indices_ = std::move(new_col_indices);

    return result;
}


template<typename T>
inline CsrMatrix<T> CsrMatrix<T>::operator-(const CsrMatrix& rhs) const
{
    if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
        throw std::invalid_argument("dimentions do not match");
    }

    CsrMatrix<T> result(rows_,cols_);

    std::vector<value_type> new_values;
    std::vector<index_type> new_col_indices;

    for (index_type r = 0; r < rows_; ++r) {
        std::unordered_map<index_type, value_type> row_map; //все элементы одной новой строки вида  (col, value)
        index_type row_start_lhs=row_ptr_[r];
        index_type row_end_lhs=row_ptr_[r+1];

        for (index_type i = row_start_lhs; i < row_end_lhs; ++i) {
            row_map[col_indices_[i]] = values_[i];
        }
        index_type row_start_rhs=rhs.row_ptr_[r];
        index_type row_end_rhs=rhs.row_ptr_[r+1];

        for (index_type i = row_start_rhs; i < row_end_rhs; ++i) {
            row_map[rhs.col_indices_[i]] -= rhs.values_[i];
        }

        std::vector<std::pair<index_type, value_type>> sorted_row(row_map.begin(), row_map.end());
        std::sort(sorted_row.begin(), sorted_row.end(), [](auto& a, auto& b){ return a.first < b.first; });

        for (auto& elem : sorted_row) {
            if (elem.second != value_type{}) {
                new_col_indices.push_back(elem.first);
                new_values.push_back(elem.second);
            }
        }

        result.row_ptr_[r + 1] = new_values.size();
    }

    result.values_ = std::move(new_values);
    result.col_indices_ = std::move(new_col_indices);

    return result;
    return CsrMatrix{};
}
template <typename T>
template <typename Scalar>
inline CsrMatrix<T> CsrMatrix<T>::operator*(Scalar scalar) const
{
    CsrMatrix<T> result(*this);
    for (index_type i=0;i<result.values_.size();++i){
        result.values_[i]*=scalar;
    }
    return result;
}
template<typename T,typename Scalar>
CsrMatrix<T> operator*(Scalar scalar, const CsrMatrix<T>& mat)
{
    return mat * scalar;
}

template<typename T>
inline typename  CsrMatrix<T>::vector_type
CsrMatrix<T>::multiply(const std::vector<value_type>& vec) const
{
    if (cols_ != vec.size() ) {
        throw std::invalid_argument("dimentions do not match");
    }
    vector_type result(rows_,value_type{});
    for (index_type row=0;row<rows_;++row){
        value_type row_result=value_type{};
        index_type row_start=row_ptr_[row];
        index_type row_end=row_ptr_[row+1];
        for (index_type i=row_start;i<row_end;++i){
            row_result+=values_[i]*vec[col_indices_[i]];
        }
        result[row]=row_result;
    }
    return result;
}

template<typename T>
inline CsrMatrix<T> CsrMatrix<T>::multiply(const CsrMatrix& rhs) const
{
    if (cols_ != rhs.rows_)
        throw std::invalid_argument("dimensions do not match");

    CsrMatrix<T> result(rows_,rhs.cols_);

    std::vector<value_type> new_values;
    std::vector<index_type> new_col_indices;

    for (index_type a_r = 0; a_r < rows_; ++a_r) {
        std::unordered_map<index_type, value_type> row_map; // (col,value) for new row

        index_type a_row_start = row_ptr_[a_r];
        index_type a_row_end   = row_ptr_[a_r + 1];

        for (index_type i = a_row_start; i < a_row_end; ++i) {
            index_type a_col = col_indices_[i];
            value_type a_val = values_[i];

            index_type b_row_start = rhs.row_ptr_[a_col];
            index_type b_row_end   = rhs.row_ptr_[a_col + 1];

            for (index_type j = b_row_start; j < b_row_end; ++j) {
                index_type b_col = rhs.col_indices_[j];
                value_type b_val = rhs.values_[j];

                row_map[b_col] += a_val * b_val;
            }
        }


        std::vector<std::pair<index_type, value_type>> sorted_row(row_map.begin(), row_map.end());
        std::sort(sorted_row.begin(), sorted_row.end(), [](auto& a, auto& b){ return a.first < b.first; });

        for (auto& elem : sorted_row) {
            if (elem.second != value_type{}) {
                new_col_indices.push_back(elem.first);
                new_values.push_back(elem.second);
            }
        }

        result.row_ptr_[a_r + 1] = new_values.size();
    }

    result.values_ = std::move(new_values);
    result.col_indices_ = std::move(new_col_indices);

    return result;
}


template<typename T>
inline CsrMatrix<T> CsrMatrix<T>::transpose() const
{
    CsrMatrix<T> result(cols_, rows_);

    std::vector<index_type> row_counts(cols_, 0); //new rows_ = cols_

    for (index_type j = 0; j < col_indices_.size(); ++j) {
        row_counts[col_indices_[j]]++;
    }

    result.row_ptr_[0] = 0;
    for (index_type i = 0; i < cols_; ++i) {
        result.row_ptr_[i + 1] = result.row_ptr_[i] + row_counts[i];
    }

    std::vector<index_type> current_pos = result.row_ptr_;

    result.values_.resize(values_.size());
    result.col_indices_.resize(col_indices_.size());

    for (index_type row = 0; row < rows_; ++row) {
        index_type row_start = row_ptr_[row];
        index_type row_end   = row_ptr_[row + 1];

        for (index_type i = row_start; i < row_end; ++i) {
            index_type col = col_indices_[i];
            value_type val = values_[i];

            index_type pos = current_pos[col]++;
            result.col_indices_[pos] = row;
            result.values_[pos] = val;
        }
    }

    return result;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const CsrMatrix<T>& mat)
{
    for (typename CsrMatrix<T>::index_type i = 0; i < mat.rows(); ++i) {
        for (typename CsrMatrix<T>::index_type j = 0; j < mat.cols(); ++j) {
            os <<std::setw(5)<< mat.at(i, j) << " ";
        }
        os << "\n";
    }
    return os;
}

template<typename T>
inline T dot(const typename CsrMatrix<T>::vector_type& a,
             const typename CsrMatrix<T>::vector_type& b)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot: vectors must be the same size");
    }

    typename CsrMatrix<T>::value_type result = T{};
    for (typename CsrMatrix<T>::index_type i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}


template<typename T>
typename CsrMatrix<T>::vector_type BiCGStab(const CsrMatrix<T>& A,
                        const typename CsrMatrix<T>::vector_type& b,
                        const typename CsrMatrix<T>::vector_type& x0 = {},
                        T tol = 1e-8,
                        size_t max_iter = 1000)
{
    using vector_type = typename CsrMatrix<T>::vector_type;
    using index_type= typename CsrMatrix<T>::index_type;
    index_type n = b.size();
    if (x0.empty() && n != A.rows())
        throw std::runtime_error("dimentions do not match");

    vector_type x = x0.empty() ? vector_type(n, T(0)) : x0;
    vector_type r = b;
    vector_type Ax = A.multiply(x);
    for (index_type i = 0; i < n; ++i) r[i] -= Ax[i];

    vector_type r_hat = r;
    vector_type p(n, T{});
    vector_type v(n, T{});

    T rho_old = T{1};
    T alpha = T{1};
    T omega = T{1};

    vector_type s(n, T{});
    vector_type t(n, T{});

    T normb = std::sqrt(dot<T>(b,b));
    if (normb == T{}) normb = 1;

    for (index_type iter = 0; iter < max_iter; ++iter) {
        T rho = dot<T>(r_hat, r);
        if (rho == T{}) throw std::runtime_error("rho=0");

        T beta = (rho / rho_old) * (alpha / omega);

        for (index_type i = 0; i < n; ++i)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);

        v = A.multiply(p);
        alpha = rho / dot<T>(r_hat, v);

        for (index_type i = 0; i < n; ++i)
            s[i] = r[i] - alpha * v[i];

        if (std::sqrt(dot<T>(s,s)) / normb < tol) {
            for (index_type i = 0; i < n; ++i)
                x[i] += alpha * p[i];
            return x;
        }

        t = A.multiply(s);
        omega = dot<T>(t,s) / dot<T>(t,t);

        for (index_type i = 0; i < n; ++i)
            x[i] += alpha * p[i] + omega * s[i];

        for (index_type i = 0; i < n; ++i)
            r[i] = s[i] - omega * t[i];

        if (std::sqrt(dot<T>(r,r)) / normb < tol)
            return x;

        if (omega == T(0)) throw std::runtime_error(" omega=0");

        rho_old = rho;
    }

    throw std::runtime_error("no converge, reached max_iter");
}
