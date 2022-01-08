#include <iostream>
#include <vector>
#include <stdio.h> /* printf, scanf, puts, NULL */
#include <random>
#include <type_traits>

using namespace std;

class Perceptron
{
public:
    // Initlize weights
    float *weights = new float[2];

    // Initilize bias
    float bias;

    // Constructor
    Perceptron()
    {
        // Generate random number between [-1, 1]
        mt19937 rng(std::random_device{}());
        uniform_real_distribution<> dist(-1, 1);

        // Generate random bias (NOT USED YET)
        bias = dist(rng);

        // initlizse random weights
        for (int i = 0; i < (sizeof(weights) / sizeof(*weights)); i++)
        {
            // Assign random number
            weights[i] = dist(rng);
            printf("Random: %f \n", weights[i]);
        }
    }

    // Simple activation function (last layer output)
    int sign(int num)
    {
        if (num <= 0)
        {
            num = -1;
        }
        else
        {
            num = 1;
        }

        return num;
    }

    // Output of first layer perceptron
    float output(float *inputs)
    {
        float sum = 0;
        for (int i = 0; i < (sizeof(weights) / sizeof(*weights)); i++)
        {

            sum += inputs[i] * weights[i];
        }

        sum = sign(sum);

        return sum;
    }
};

// Simple matrix class for
class Matrix
{
private:
    int rows_, cols_;
    float **matrix_;

public:
    // Initlize matrix with zeros
    Matrix(int rows, int cols) : rows_(rows), cols_(cols)
    {
        // Allocate space for matrix
        matrix_ = new float *[rows_];
        for (int i = 0; i < rows_; i++)
        {
            matrix_[i] = new float[cols_];
        }

        // Assign zeros to matrix
        for (int i = 0; i < rows_; i++)
        {
            for (int j = 0; j < cols_; j++)
            {
                matrix_[i][j] = 0.0;
            }
        }
    }

    // use this function to scale our matrix
    void multiply(float num)
    {
        for (int i = 0; i < rows_; i++)
        {
            for (int j = 0; j < cols_; j++)
            {
                matrix_[i][j] = matrix_[i][j] * num;
            }
        }
    }



    // Used to add a number to each element in our matrix
    template <typename T>
    void add(T num)
    {
       
        //If type is a single number
        if(std::is_same<T, float>::value)
        {
            for (int i = 0; i < rows_; i++)
            {
                for (int j = 0; j < cols_; j++)
                {
                    matrix_[i][j] = matrix_[i][j] + num;
                }
            }
        }

         // If type is a matrix
        else if (std::is_same<T, Matrix>::value)
        {
            cout << "hallo" << endl;
            for (int i = 0; i < rows_; i++)
            {
                for (int j = 0; j < cols_; j++)
                {
                    matrix_[i][j] = matrix_[i][j] + num.matrix_[i][j];
                }
            }

        }
        else 
        {
            exit(1);
        }
    }

    // Print matrix
    void print()
    {
        for (int i = 0; i < rows_; i++)
        {
            for (int j = 0; j < cols_; j++)
            {
                cout << matrix_[i][j] << " ";
            }
            cout << endl;
        }
    }
};

int main()
{

    // // Test M
    // Perceptron test1;

    // float inputs[2] = {0.5, -0.2};

    // printf("Output %f \n", test1.output(inputs));

    // std::cout << "hello world" << std::endl;

    Matrix matrix1(2, 2);

    Matrix matrix2(2, 2);

    float num = 2.0;

    matrix1.add<float>(num);

    matrix1.print();
}