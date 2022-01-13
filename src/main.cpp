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
/*
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

*/








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
    static Matrix list_to_matrix(float *list, int size)
    {

        // Turn list of input into matrix
        Matrix input_matrix(size, 1);



        for(int i = 0; i < size; i++)
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
                    throw runtime_error("Matrix product not possible!");
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
            cout << "Type of variable is: " << typeid(num).name() << endl;
        }
      
        
    }



     // Used to add a number to each element in our matrix
    
    Matrix subtract(Matrix num)
    {
        try {
       
           

            // If type is a matrix
            if constexpr (std::is_same<Matrix, Matrix>::value)
            {
                Matrix result(rows_, cols_);


                for (int i = 0; i < rows_; i++)
                {
                    for (int j = 0; j < cols_; j++)
                    {
                        result.matrix_[i][j] = matrix_[i][j] - num.matrix_[i][j];
                    }
                }

                return result;

            }
            else 
            {
                throw(num);
            }

        }

        catch(Matrix num)
        {
            cout << "Type is undefined - Must be matrix" << endl;
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



    // Activation function
    float sigmoid(float x)
    {
        return 1 / (1+ exp(-x));

    }


    // Apply activation function to all neuron of matrix
    void generate_output()
    {
        for(int i = 0; i < rows_; i++)
        {
            for(int j= 0; j < cols_; j++)
            {
                matrix_[i][j] = sigmoid(matrix_[i][j]);
            }
        }

    }


    float *to_list()
    {
        float outputs[rows_ * cols_];

        for(int i = 0; i < rows_; i++)
        {
            for(int j = 0; j < cols_; j++)
            {
                outputs[i + j] = matrix_[i][j];
            }
        }

        return outputs;

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

        

        

        Matrix forward_pass(float *input_list, int size)
        {
            // Generate input matrix
            Matrix inputs = Matrix::list_to_matrix(input_list, size);


            // Generate hidden layer
            Matrix hidden_layer = weights_input_hidden->multiply(inputs);

            // Add bias to each neuron in hidden layer
            hidden_layer.add(*bias_hidden);

            // Output of hidden layer
            hidden_layer.generate_output();

            // Generate final output
            Matrix output = weights_hidden_output->multiply(hidden_layer);
            
            // Add bias
            output.add(*bias_output);

            // Final result (last neuron)
            output.generate_output(); 

            
            return output;


        }

        void backward_pass()
        {

        }


        void train(float *inputs_list, int input_size, float *targets_list, int target_size)
        {
            // Input matrix
            Matrix outputs = this->forward_pass(inputs_list, input_size);
            
            // Target matrix
            Matrix targets = Matrix::list_to_matrix(targets_list, target_size);

        
            // Output layer error (ERROR = TARGET - OUTPUTS)
            Matrix output_errors = targets.subtract(outputs);

            // Hidden layer error
            Matrix weight_hidden_output_tr = weights_hidden_output->transpose();
            Matrix hidden_errors = weight_hidden_output_tr.multiply(output_errors);

            cout << "Target: " << endl;
            targets.print();

            cout << "Outputs: " << endl;
            outputs.print();

            cout << "Error: " << endl;
            output_errors.print();
            

        }
};



int main()
{

   
    
    float inputs[] = {1.0, 0.0};
    float targets[] = {1.0};

    NeuralNetwork *nn = new NeuralNetwork(2,2,1);

    nn->train(inputs, 2, targets, 1);

    //nn->forward_pass(input);



}