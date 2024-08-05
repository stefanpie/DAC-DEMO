#include "./model.h"

const int dim_in = INPUT_DIM;
const int dim_out = OUTPUT_DIM;
const int dim_hidden = HIDDEN_DIM;
const int num_hidden_layers = NUM_HIDDEN_LAYERS;

float input_layer_weight[dim_hidden][dim_in];
float input_layer_bias[dim_hidden];
float input_layer_w0[1];
float hidden_layers_weight[num_hidden_layers][dim_hidden][dim_hidden];
float hidden_layers_bias[num_hidden_layers][dim_hidden];
float hidden_layers_w0[num_hidden_layers];
float output_layer_weight[dim_out][dim_hidden];
float output_layer_bias[dim_out];

F_TYPE input_layer_weight_fixed[dim_hidden][dim_in];
F_TYPE input_layer_bias_fixed[dim_hidden];
F_TYPE input_layer_w0_fixed[1];
F_TYPE hidden_layers_weight_fixed[num_hidden_layers][dim_hidden][dim_hidden];
F_TYPE hidden_layers_bias_fixed[num_hidden_layers][dim_hidden];
F_TYPE hidden_layers_w0_fixed[num_hidden_layers];
F_TYPE output_layer_weight_fixed[dim_out][dim_hidden];
F_TYPE output_layer_bias_fixed[dim_out];

int main() {



    load_data_2d<dim_hidden, dim_in>("test_data/siren_input_layer_weight.bin", input_layer_weight);
    load_data_1d<dim_hidden>("test_data/siren_input_layer_bias.bin", input_layer_bias);
    load_data_1d<1>("test_data/siren_input_layer_w0.bin", input_layer_w0);
    load_data_3d<num_hidden_layers, dim_hidden, dim_hidden>("test_data/siren_hidden_layers_weight.bin", hidden_layers_weight);
    load_data_2d<num_hidden_layers, dim_hidden>("test_data/siren_hidden_layers_bias.bin", hidden_layers_bias);
    load_data_1d<num_hidden_layers>("test_data/siren_hidden_layers_w0.bin", hidden_layers_w0);
    load_data_2d<dim_out, dim_hidden>("test_data/siren_output_layer_weight.bin", output_layer_weight);
    load_data_1d<dim_out>("test_data/siren_output_layer_bias.bin", output_layer_bias);



    cast_2d<dim_hidden, dim_in, float, F_TYPE>(input_layer_weight, input_layer_weight_fixed);
    cast_1d<dim_hidden, float, F_TYPE>(input_layer_bias, input_layer_bias_fixed);
    cast_1d<1, float, F_TYPE>(input_layer_w0, input_layer_w0_fixed);
    cast_3d<num_hidden_layers, dim_hidden, dim_hidden, float, F_TYPE>(hidden_layers_weight, hidden_layers_weight_fixed);
    cast_2d<num_hidden_layers, dim_hidden, float, F_TYPE>(hidden_layers_bias, hidden_layers_bias_fixed);
    cast_1d<num_hidden_layers, float, F_TYPE>(hidden_layers_w0, hidden_layers_w0_fixed);
    cast_2d<dim_out, dim_hidden, float, F_TYPE>(output_layer_weight, output_layer_weight_fixed);
    cast_1d<dim_out, float, F_TYPE>(output_layer_bias, output_layer_bias_fixed);

    // model input
    float x_in[dim_in];
    F_TYPE x_in_fixed[dim_in];
    load_data_1d<dim_in>("test_data/siren_x_in.bin", x_in);
    cast_1d<dim_in, float, F_TYPE>(x_in, x_in_fixed);

    // golden output
    float x_out_golden[dim_out];
    load_data_1d<dim_out>("test_data/siren_x_out.bin", x_out_golden);

    // kernel
    float x_out_kernel_float[dim_out];
    F_TYPE x_out_kernel_fixed[dim_out];

    // run kernel twice to load weights then run inference
    siren_test_model(
        x_in_fixed,
        x_out_kernel_fixed,
        1,
        input_layer_weight_fixed,
        input_layer_bias_fixed,
        input_layer_w0_fixed,
        hidden_layers_weight_fixed,
        hidden_layers_bias_fixed,
        hidden_layers_w0_fixed,
        output_layer_weight_fixed,
        output_layer_bias_fixed
    );

    siren_test_model(
        x_in_fixed,
        x_out_kernel_fixed,
        0,
        input_layer_weight_fixed,
        input_layer_bias_fixed,
        input_layer_w0_fixed,
        hidden_layers_weight_fixed,
        hidden_layers_bias_fixed,
        hidden_layers_w0_fixed,
        output_layer_weight_fixed,
        output_layer_bias_fixed
    );

    cast_1d<dim_out, F_TYPE, float>(x_out_kernel_fixed, x_out_kernel_float);

    // compare
    bool pass = true;
    float test_delta = 1e-3;
    for (int i = 0; i < dim_out; i++) {
        float diff = fabs(x_out_kernel_float[i] - x_out_golden[i]);
        if (diff > test_delta) {
            pass = false;
        }
    }

    if (pass) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    print_array_1d<dim_out, float>(x_out_golden);
    print_array_1d<dim_out, float>(x_out_kernel_float);

}