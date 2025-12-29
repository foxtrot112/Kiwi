#pragma once


#include <fstream>
#include <vector>
#include <string>
#include "tensor.cc"
#include <filesystem>

// Save 2D tensor
void saveTensor2Bin(const tensor2 &tensor, const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    if(!out) {
        std::cerr << "Error opening file for writing: " << filename << "\n";
        return;
    }

    size_t rows = tensor.size();
    size_t cols = tensor[0].size();
    out.write((char*)&rows, sizeof(size_t));
    out.write((char*)&cols, sizeof(size_t));

    for(size_t i=0;i<rows;i++)
        out.write((char*)tensor[i].data(), cols * sizeof(float));

    out.close();
}

// Save 1D tensor
void saveTensor1Bin(const tensor1 &tensor, const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    if(!out) { std::cerr << "Error opening file: " << filename << "\n"; return; }
    size_t size = tensor.size();
    out.write((char*)&size, sizeof(size_t));
    out.write((char*)tensor.data(), size * sizeof(float));
    out.close();
}

tensor2 loadTensor2Bin(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if(!in) { std::cerr << "File not found: " << filename << "\n"; return tensor2(); }

    size_t rows, cols;
    in.read((char*)&rows, sizeof(size_t));
    in.read((char*)&cols, sizeof(size_t));

    tensor2 tensor(rows, tensor1(cols));
    for(size_t i=0;i<rows;i++)
        in.read((char*)tensor[i].data(), cols * sizeof(float));

    in.close();
    return tensor;
}

tensor1 loadTensor1Bin(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if(!in) { std::cerr << "File not found: " << filename << "\n"; return tensor1(); }

    size_t size;
    in.read((char*)&size, sizeof(size_t));
    tensor1 tensor(size);
    in.read((char*)tensor.data(), size * sizeof(float));
    in.close();
    return tensor;
}



bool fileExists(const std::string &filename) {
    return std::filesystem::exists(filename);
}

