#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include "utils.h"

////////////////////////////////////////////////////////////////////
// Series Class
Series::Series(int length) {
    // Length of the series
    this->length = length;
    // Allocate memory for the values
    this->values = (double *) malloc(sizeof(double) * length);
    // Initialize the values to 0
    for(int i = 0; i < length; ++i)
        this->values[i] = 0;
}

void Series::setValues(double* values) {
    // Copy the values to the series
    for (int i = 0; i < length; i++) {
        this->values[i] = values[i];
    }
}

void Series::print(int decimals) {
    // Print the length of the series
    std::cout << "Series with " << this->length << " elements:\n";

    // Print single column header
    std::cout << "\t0:\t\n";

    // Iterate over the elements
    for (int i = 0; i < this->length; i++) {
        // If more than 5 elements,
        // only print first and last 5
        if(i == 5 && this->length > 10) {
            std::cout << "...\t...\n";
            // Skip to the last 5 elements
            i = this->length - 5;
        }
        // Print the element
        std::cout << i 
          // Tab character
          << ":\t" 
          // Set fixed output format
          << std::fixed 
          // Set precision for decimals
          << std::setprecision(decimals) 
          // Output the value
          << this->values[i] 
          // New line
          << "\n";
    }        
    // New line
    std::cout << "\n";                                   
}

Series::~Series() {
    // Deallocate the memory
    free(this->values);
}

////////////////////////////////////////////////////////////////////
// DataFrame Class

// Big Changes: DataFrame and Series classes are separate
// DataFrame is now going to use a 2D Array of doubles

DataFrame::DataFrame(int numRows, int numCols) {
        this->numRows = numRows;
        this->numCols = numCols;

        // Allocate Memory for a 2D Array
        values = (double **) malloc(sizeof(double *) * numCols);
        // Iterate over the columns
        for (int i = 0; i < numCols; i++) {
            // Allocate memory for each column
            values[i] = (double *) malloc(sizeof(double) * numRows);
            // Initialize the values to 0
            for (int j = 0; j < numRows; j++) {
                values[i][j] = 0;
            }
        }
}

void DataFrame::setValues(double** values) {
    for (int i = 0; i < this->numCols; i++) {
        for (int j = 0; j < this->numRows; j++) {
            this->values[i][j] = values[i][j];
        }
    }
}

void DataFrame::read_csv(const std::string &filename) {
    // Open the file
    std::ifstream file(filename);
    // String for the line
    std::string line;
    // String for the cell
    std::string cell;
    // Row Counter
    int row = 0;
    // Loop over the rows
    while (std::getline(file, line)) {
        // Create a stringstream from the line
        std::stringstream lineStream(line);
        // Column Counter
        int col = 0;
        // Loop over the columns
        while (std::getline(lineStream, cell, ',')) {
            // Convert the cell to a double
            values[col][row] = std::stod(cell);
            // Increment the column counter
            col++;
        }
        row++;
    }
    file.close();
}

void DataFrame::read_csv_limited(const std::string& filename,
                                 int startRow,
                                 int endRow) {
    // Open the file
    std::ifstream file(filename);
    // String for the line
    std::string line;
    // String for the cell
    std::string cell;
    // Row Counter
    int row = 0;

    // Loop over the rows
    while (std::getline(file, line)) {
        // Check if row is within range
        if (row >= startRow && row < endRow) {
            // Create a stringstream from the line
            std::stringstream lineStream(line);
            // Column Counter
            int col = 0;
            // Loop over the columns
            while (std::getline(lineStream, cell, ',')) {
                // Convert the cell to double
                values[col][row - startRow] = std::stod(cell);
                // Increment the column counter
                col++;
            }
        }
        row++;
        // Stop reading
        if (row >= endRow) break;
    }
    file.close();
}

void DataFrame::print(int decimals) {
    // Introduce the DataFrame details
    std::cout << "DataFrame with ";
    std::cout << numRows;
    std::cout << " rows and ";
    std::cout << numCols;
    std::cout << " columns:\n";
    
    // Begin printing column headers
    std::cout << "\t";
    for (int j = 0; j < numCols; j++) {
        // Apply ellipsis
        if (j == 5 && numCols > 10) {
            std::cout << "...\t";
            j = numCols - 5;
        }
        // Print column index
        std::cout << j << ":\t";
    }
    // End line for column headers
    std::cout << std::endl;

    // Iterate through each row
    for (int i = 0; i < numRows; i++) {
        // Apply ellipsis
        if (i == 5 && numRows > 10) {
            std::cout << "...\t";
            for (int k = 0; k < numCols; k++) {
                // Apply ellipsis
                if (k == 5 && numCols > 10) {
                    std::cout << "...\t";
                    k = numCols - 5;
                }
                // Apply ellipsis
                std::cout << "...\t";
            }
            // End line for the current row
            std::cout << std::endl;
            i = numRows - 5;
        }

        // Print row index
        std::cout << i << ":\t";
        for (int j = 0; j < numCols; j++) {
            // Apply ellipsis
            if (j == 5 && numCols > 10) {
                std::cout << "...\t";
                j = numCols - 5;
            }
            // Set fixed output format
            std::cout << std::fixed;
            // Set precision for decimals
            std::cout << std::setprecision(decimals);
            // Output the matrix value
            std::cout << values[j][i];
            // Add tab for column separation
            std::cout << "\t";
        }
        // End line for the current row
        std::cout << std::endl;
    }
    // Add a new line
    std::cout << std::endl;
}

DataFrame DataFrame::transpose() {
    // Initialize result DataFrame
    DataFrame result(numCols, numRows);
    // Loop through rows
    for (int i = 0; i < numRows; i++) {
        // Loop through columns
        for (int j = 0; j < numCols; j++) {
            // Transpose the values
            result.values[i][j] = this->values[j][i];
        }
    }
    return result;
}

DataFrame::~DataFrame() {
    // Free the memory for each column
    for (int i = 0; i < numCols; i++) {
        free(values[i]);
    }
    // Free the memory for the 2D Array
    free(values);
}
