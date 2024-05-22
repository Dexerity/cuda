#pragma once

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>

using namespace cv;

class Stack
{
    private:
        Mat* data;
        int capacity;
        int top;

    public:
        Stack(int size) {
            capacity = size + 1;
            data = new Mat[capacity];
            top = -1;
        }

        ~Stack() {
            delete[] data;
        }

        bool isEmpty() {
            return top == -1;
        }

        bool isFull() {
            return top == capacity - 1;
        }

        void push(Mat element) {
            if (isFull()) {
                shift();
                top--;
            }
            top++;
            data[top] = element;
        }

        Mat pop() {
            if (isEmpty()) {
                printf("Stack Underflow\n");
                return Mat();
            }

            Mat element = data[0];
            shift();
            return element;
        }

        Mat peek() {
            if (isEmpty()) {
                printf("Stack Underflow\n");
                return Mat();
            }

            return data[0];
        }

        void shift() {
            for (int i = 0; i < top; i++) {
                data[i] = data[i + 1];
            }
            top--;
        }

        void print() {
            for (int i = 0; i <= top; i++) {
                printf("%d ", data[i].channels());
            }
            printf("\n");
        }
};