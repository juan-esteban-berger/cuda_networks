/**
 * @file vector.h
 * @brief Defines the Vector class for GPU-accelerated vector operations.
 */

#ifndef VECTOR_H
#define VECTOR_H

/**
 * @class Vector
 * @brief Represents a vector with GPU-accelerated operations.
 */
class Vector {
public:
    /**
     * @brief Construct a new Vector object
     * @param rows Number of elements in the vector
     */
    Vector(int rows);

    /**
     * @brief Destroy the Vector object
     */
    ~Vector();

    /**
     * @brief Initialize the vector (typically sets all elements to zero)
     */
    void initialize();

    /**
     * @brief Randomize the vector elements with values between -0.5 and 0.5
     */
    void randomize();

    /**
     * @brief Print the vector contents
     * @param decimals Number of decimal places to display
     */
    void print(int decimals);

    /**
     * @brief Get the number of elements in the vector
     * @return int Number of elements
     */
    int get_rows() const;

    /**
     * @brief Get the raw data pointer of the vector
     * @return double* Pointer to the vector data on the device
     */
    double* get_data() const;

    /**
     * @brief Creates a deep copy of the vector.
     * @return A new Vector object with the same content as the original.
     */
    Vector copy() const;

    /**
     * @brief Multiplies all elements in the vector by a scalar.
     * @param scalar The scalar to multiply by.
     */
    void multiply_scalar(double scalar);
    
    /**
     * @brief Divides all elements in the vector by a scalar.
     * @param scalar The scalar to divide by.
     * @throws std::invalid_argument if scalar is exactly zero.
     */
    void divide_scalar(double scalar);

private:
    int rows;    ///< Number of elements in the vector
    double* d_data;  ///< Device data pointer
};

#endif // VECTOR_H
