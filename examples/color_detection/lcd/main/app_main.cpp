#include "who_camera.h"
#include "who_color_detection.hpp"
#include "custom_lcd.h"
#include "who_button.h"
#include "event_logic.hpp"
#include "who_adc_button.h"

#include <stdio.h>
#include <stdlib.h>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "dl_tool.hpp"
#include "custom_coefficient_model.hpp"

static QueueHandle_t xQueueAIFrame = NULL;
static QueueHandle_t xQueueLCDFrame = NULL;
//static QueueHandle_t xQueueADCKeyState = NULL;
//static QueueHandle_t xQueueGPIOKeyState = NULL;
//static QueueHandle_t xQueueEventLogic = NULL;
//static button_adc_config_t buttons[4] = {{1, 2800, 3000}, {2, 2250, 2450}, {3, 300, 500}, {4, 850, 1050}};

#define GPIO_BOOT GPIO_NUM_0

extern "C" void app_main()
{
    int input_height = 120;
    int input_width = 120;
    int input_channel = 3;
    int input_exponent = -7;
    int8_t *model_input = (int8_t *)dl::tool::malloc_aligned_prefer(input_height*input_width*input_channel, sizeof(int8_t));
    

    Tensor<int8_t> input;
    input.set_element((int8_t *)model_input).set_exponent(input_exponent).set_shape({120, 120, 3}).set_auto_free(false);

    Custom model;

    dl::tool::Latency latency;


    gpio_config_t gpio_conf;
    gpio_conf.mode = GPIO_MODE_OUTPUT_OD;
    gpio_conf.intr_type = GPIO_INTR_DISABLE;
    gpio_conf.pin_bit_mask = 1LL << GPIO_NUM_3;
    gpio_config(&gpio_conf);
    
    xQueueAIFrame = xQueueCreate(2, sizeof(camera_fb_t *));
    xQueueLCDFrame = xQueueCreate(2, sizeof(camera_fb_t *));
    //xQueueADCKeyState = xQueueCreate(1, sizeof(int));
    //xQueueGPIOKeyState = xQueueCreate(1, sizeof(int));
    //xQueueEventLogic = xQueueCreate(1, sizeof(int));

    register_camera(PIXFORMAT_RGB565, FRAMESIZE_240X240, 2, xQueueLCDFrame);
    //register_adc_button(buttons, 4, xQueueADCKeyState);
    //register_button(GPIO_NUM_0, xQueueGPIOKeyState);
    //register_event(xQueueADCKeyState, xQueueGPIOKeyState, xQueueEventLogic);
    //register_color_detection(xQueueAIFrame, xQueueEventLogic, NULL, xQueueLCDFrame, false);
    register_custom_lcd(xQueueLCDFrame, NULL, true);

    camera_fb_t *frame = NULL;
    while(true)
    {
        if (xQueueReceive(xQueueAIFrame, &frame, portMAX_DELAY))
        {
            for (unsigned int y = 0; y < frame->height/2; y++)
            {
                for (unsigned int x = 0; x < frame->width/2; x++)
                {
                    //((uint16_t*)frame->buf)[y * frame->width + x] = ((uint16_t*)frame->buf)[y * frame->width + x] | 0b1110000000000011;
                    float r = ((float)((((uint16_t*)frame->buf)[y * 2 * frame->width + x * 2] >> 3) & 0b11111)) / 31.0;
                    float b = ((float)((((uint16_t*)frame->buf)[y * 2 * frame->width + x * 2] >> 8) & 0b11111)) / 31.0;
                    float g = ((float)(((((uint16_t*)frame->buf)[y * 2 * frame->width + x * 2] << 3 ) & 0b111000) | ((((uint16_t*)frame->buf)[y * frame->width + x] >> 13 ) & 0b111))) / 63.0;
                    model_input[y * input_width * input_channel + x * input_channel + 0] = (int8_t)DL_CLIP(r * (float)(1 << -input_exponent), -128, 127);
                    model_input[y * input_width * input_channel + x * input_channel + 1] = (int8_t)DL_CLIP(g * (float)(1 << -input_exponent), -128, 127);
                    model_input[y * input_width * input_channel + x * input_channel + 2] = (int8_t)DL_CLIP(b * (float)(1 << -input_exponent), -128, 127);
                }
                
            }
            // model forward
            // latency.start();
            model.forward(input);
            // latency.end();

            // parse
            int8_t *score = model.l14.get_output().get_element_ptr();
            uint16_t box = 0b1110000000000111;
            //printf("%d, %d\n", (int8_t)score[0], (int8_t)score[1]);
            if (score[0] > score[1])
            {
                box = 0b0000000011111000;
            }
            for (unsigned int i = 0; i < frame->width * 24; i++)
            {
                ((uint16_t *)frame->buf)[i] = box;
            }
            
            xQueueSend(xQueueLCDFrame, &frame, portMAX_DELAY);
        }
    }

}
