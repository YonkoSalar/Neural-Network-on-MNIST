#include <iostream>
#include <vector>
#include <stdio.h> /* printf, scanf, puts, NULL */
#include <random>
#include <type_traits>
#include <stdexcept>
#include <bits/stdc++.h>



using namespace std;


/////////////////
// PERCEPTRON //
////////////////

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

    

    // Output of first layer perceptron
    float output(float *inputs)
    {
        float sum = 0;
        for (int i = 0; i < (sizeof(weights) / sizeof(*weights)); i++)
        {

            sum += (inputs[i] * weights[i]);
        }

        sum = sigmoid(sum);

        return sum;
    }
};




////////////////////////
//  SIGMOID FUNCTION //
///////////////////////
float sigmoid(float x)
{
    return 1 / (1+ exp(-x));

}



//////////////////////
//      MATRIX     //
/////////////////////
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

    

    // We can call this function without creating object
    static Matrix list_to_matrix(float *list)
    {

        // Turn list of input into matrix
        Matrix input_matrix(sizeof(list) / sizeof(*list)+1, 1);


        for(int i = 0; i < sizeof(list) / sizeof(*list); i++)
        {
            input_matrix.matrix_[i][0] = list[i];
        }

        
        return input_matrix;

    }


    void randomize()
    {
         // Generate random number between [-1, 1]
        mt19937 rng(std::random_device{}());
        uniform_int_distribution<> dist(-1, 1);


        for(int i = 0; i < rows_; i++)
        {
            for(int j = 0; j < cols_; j++)
            {
                matrix_[i][j] = dist(rng);
            }
        }


    }



    // use this function to scale our matrix
     template <typename T>
    T multiply(T num)
    {

        try 
        {
             // Matrix product
            if constexpr (std::is_same<T, Matrix>::value)
            {
                // Chekc if matix product is possible
                if(cols_ != num.rows_)
                {
                    throw runtime_error("Matrid product not possible!");
                }
                
                // New Matrix
                Matrix result = Matrix(rows_, num.cols_);

                // Matrix product
                for(int i = 0; i < result.rows_; i++)
                {
                    for(int j = 0; j < result.cols_; j++)
                    {
                        
                        for(int k = 0; k < rows_; k++)
                        {
                            result.matrix_[i][j] += matrix_[i][k] * num.matrix_[k][j];
                        }
                    }
                }

                return result;




            }
            // Scalar
            else if constexpr (std::is_same<T, float>::value)
            {
                
                for (int i = 0; i < rows_; i++)
                {
                    for (int j = 0; j < cols_; j++)
                    {
                        matrix_[i][j] = matrix_[i][j] * num;
                    }
                }

            }
            else 
            {
                throw runtime_error("Undefined type");
            }

        }

        catch (const std::exception &ex)
        {
            cout << ex.what() << "\n";
        }

       
        
        
    }



    // Used to add a number to each element in our matrix
    template <typename T>
    void add(T num)
    {
        try {
       
            //If type is a single number
            if constexpr (std::is_same<T, float>::value)
            {
                cout << "inside if" << endl;
                for (int i = 0; i < rows_; i++)
                {
                    for (int j = 0; j < cols_; j++)
                    {
                        matrix_[i][j] = matrix_[i][j] + num;
                    }
                }
            }

            // If type is a matrix
            else if constexpr (std::is_same<T, Matrix>::value)
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
                throw(num);
            }

        }

        catch(T num)
        {
            cout << "Type is undefined - Must be either matrix or float" << endl;
            cout << "Type of variable num is: " << typeid(num).name() << endl;
        }
      
        
    }

    Matrix transpose()
    {
        Matrix result = Matrix(cols_, rows_);

        for(int i = 0; i < rows_; i++)
        {
            for(int j = 0; j < cols_; j++)
            {
                result.matrix_[j][i] = matrix_[i][j];
            }
        }

        return result;

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





//////////////////////
//  NEURAL NETWORK //
/////////////////////


class NeuralNetwork 
{
    private:
        int input_nodes;
        int hidden_nodes;
        int output_nodes;


        Matrix *weights_input_hidden;
        Matrix *weights_hidden_output;

        Matrix *bias_hidden;
        Matrix *bias_output;




    public: 
        NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) 
        : input_nodes(input_nodes), hidden_nodes(hidden_nodes), output_nodes(output_nodes)
        {
            // Weight matrix from input to hidden layer
            weights_input_hidden = new Matrix(hidden_nodes, input_nodes);   

            // Weight matrix from hidden to output layer
            weights_hidden_output = new Matrix(output_nodes, hidden_nodes);

            // Randomize weights
            weights_input_hidden->randomize();
            weights_hidden_output->randomize();

            // Matrix for biases which is number of nodes 
            bias_hidden = new Matrix(hidden_nodes, 1);
            bias_output = new Matrix(output_nodes, 1);



        }

        



        

        //template <typename T>
        void forward_pass(float *input_list)
        {
            // Generate input matrix
            Matrix inputs = Matrix::list_to_matrix(input_list);

            // Generate hidden layer
            Matrix hidden_layer = weights_input_hidden->multiply(inputs);

            // Add bias to each hidden neuron
            hidden_layer.add(bias_hidden);

            


        }

        void backward_pass()
        {

        }

};



int main()
{

    // // Test M
    // Perceptron test1;

    // float inputs[2] = {0.5, -0.2};

    // printf("Output %f \n", test1.output(inputs));

    // std::cout << "hello world" << std::endl;

    Matrix matrix1(3,3);

    matrix1.randomize();
    //matrix1.print();

    float input[] = {1.0, 2.0, 1.2};

    cout << "Size main: " << sizeof(input) / sizeof(*input) << endl;
    Matrix input_matrix = Matrix::list_to_matrix(input);

    input_matrix.print();



    

    /*
    Matrix matrix2(2, 2);

    matrix1.add<float>(5.0);
    matrix2.add<float>(2.0);

    

    matrix1.print();
    matrix2.print();


    Matrix matrix3 = matrix1.multiply(matrix2);

    matrix3.print();
    */


}