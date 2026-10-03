/*****************************************
 * C Program for Linear Regression
 *****************************************/

#include <stdio.h>

// Function to calculate mean
double mean(double arr[], int size) 
{
    double sum = 0;
    for (int i = 0; i < size; i++) 
    {
        sum += arr[i];
    }
    return sum / size;
}

// Function to calculate Linear Regression coefficients (y = m*x + b)
void linearRegression(double x[], double y[], int size, double *m, double *b) 
{
    double x_mean = mean(x, size);
    double y_mean = mean(y, size);
    
    double numerator = 0, denominator = 0;
    
    for (int i = 0; i < size; i++) {
        numerator += (x[i] - x_mean) * (y[i] - y_mean);
        denominator += (x[i] - x_mean) * (x[i] - x_mean);
    }
    
    *m = numerator / denominator; // Slope
    *b = y_mean - (*m * x_mean);  // Intercept
}

int main() 
{
    // Sample dataset: Square footage (x) and Price in $1000s (y)
    double x[] = {1400, 1600, 1700, 1875, 1100};
    double y[] = {245, 312, 279, 308, 199};
    int size = sizeof(x) / sizeof(x[0]);

    double m, b;
    linearRegression(x, y, size, &m, &b);
    
    printf("Linear Regression Model: y = %.2fx + %.2f\n", m, b);
    
    // Predict price for a new house of 1500 sq. ft.
    double new_x = 1500;
    double predicted_y = m * new_x + b;
    printf("Predicted price for 1500 sq. ft. house: $%.2fK\n", predicted_y);

    return 0;
}
/**************************************************************************/
