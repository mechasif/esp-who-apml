#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "dl_layer_model.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_transpose.hpp"
#include "dl_layer_softmax.hpp"
#include "dl_layer_sigmoid.hpp"
#include "custom_coefficient.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace custom_coefficient;

class Custom : public Model<int8_t> // Derive the Model class in "dl_layer_model.hpp"
{
private:
    // Declare layers as member variables
    Conv2D<int8_t> l1;
    MaxPool2D<int8_t> l2;
    Conv2D<int8_t> l3;
    MaxPool2D<int8_t> l4;
    Conv2D<int8_t> l5;
    Conv2D<int8_t> l6;
    Conv2D<int8_t> l7;
    MaxPool2D<int8_t> l8;
    Transpose<int8_t> l9;
    Reshape<int8_t> l10;
    Conv2D<int8_t> l11;
    Conv2D<int8_t> l12;
    Conv2D<int8_t> l13;
    

public:
    Sigmoid<int8_t, int8_t, QIQO, false> l14;

    /**
     * @brief Initialize layers in constructor function
     * 
     */
    Custom() : //l1(Transpose<int8_t>({}, "l0")),
              l1(Conv2D<int8_t>(-6, get_sequential_conv2d_biasadd_filter(), get_sequential_conv2d_biasadd_bias(), NULL, PADDING_VALID, {}, 4, 4, "l1")),
              l2(MaxPool2D<int8_t>({3,3}, PADDING_VALID, {}, 2, 2, "l2")),
              l3(Conv2D<int8_t>(-5, get_sequential_conv2d_1_biasadd_filter(), get_sequential_conv2d_1_biasadd_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l3")),
              l4(MaxPool2D<int8_t>({3, 3}, PADDING_VALID, {}, 2, 2, "l4")),
              l5(Conv2D<int8_t>(-4, get_sequential_conv2d_2_biasadd_filter(), get_sequential_conv2d_2_biasadd_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l5")),
              l6(Conv2D<int8_t>(-3, get_sequential_conv2d_3_biasadd_filter(), get_sequential_conv2d_3_biasadd_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l6")),
              l7(Conv2D<int8_t>(-2, get_sequential_conv2d_4_biasadd_filter(), get_sequential_conv2d_4_biasadd_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l7")),
              l8(MaxPool2D<int8_t>({3, 3}, PADDING_VALID, {}, 2, 2, "l8")),
              l9(Transpose<int8_t>({}, "l9")),
              l10(Reshape<int8_t>({1, 1, 256}, "l10")),
              l11(Conv2D<int8_t>(-1, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_SAME_END, {}, 1, 1, "l11")),
              l12(Conv2D<int8_t>(-1, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), get_fused_gemm_1_activation(), PADDING_SAME_END, {}, 1, 1, "l12")),
              l13(Conv2D<int8_t>(-2, get_fused_gemm_2_filter(), get_fused_gemm_2_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l15")),
              l14(Sigmoid<int8_t, int8_t, QIQO, false>(-6, "l16")){}
    /**
     * @brief call each layers' build(...) function in sequence
     * 
     * @param input 
     */
    void build(Tensor<int8_t> &input)
    {
        //this->l1.build(input);
        //printf("%d", this->l1.get_output().shape[2]);
        this->l1.build(input);
        this->l2.build(this->l1.get_output());
        this->l3.build(this->l2.get_output());
        this->l4.build(this->l3.get_output());
        this->l5.build(this->l4.get_output());
        this->l6.build(this->l5.get_output());
        this->l7.build(this->l6.get_output());
        this->l8.build(this->l7.get_output());
        this->l9.build(this->l8.get_output());
        this->l10.build(this->l9.get_output());
        this->l11.build(this->l10.get_output());
        this->l12.build(this->l11.get_output());
        this->l13.build(this->l12.get_output());
        this->l14.build(this->l13.get_output());
    }

    /**
     * @brief call each layers' call(...) function in sequence
     * 
     * @param input 
     */
    void call(Tensor<int8_t> &input)
    {
        this->l1.call(input);
        input.free_element();

        this->l2.call(this->l1.get_output());
        this->l1.get_output().free_element();

        this->l3.call(this->l2.get_output());
        this->l2.get_output().free_element();

        this->l4.call(this->l3.get_output());
        this->l3.get_output().free_element();
        
        this->l5.call(this->l4.get_output());
        this->l4.get_output().free_element();
        
        this->l6.call(this->l5.get_output());
        this->l5.get_output().free_element();
        
        this->l7.call(this->l6.get_output());
        this->l6.get_output().free_element();
        
        this->l8.call(this->l7.get_output());
        this->l7.get_output().free_element();
        
        this->l9.call(this->l8.get_output());
        this->l8.get_output().free_element();
        
        this->l10.call(this->l9.get_output());
        this->l9.get_output().free_element();
        
        this->l11.call(this->l10.get_output());
        this->l10.get_output().free_element();
        
        this->l12.call(this->l11.get_output());
        this->l11.get_output().free_element();
        
        this->l13.call(this->l12.get_output());
        this->l12.get_output().free_element();

        this->l14.call(this->l13.get_output());
        this->l13.get_output().free_element();
    }
};
