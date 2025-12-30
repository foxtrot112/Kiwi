#pragma once

#include <iostream>
#include <string>
#include "tensor.cc"

#include <unordered_set>
#include <stdexcept>
#include <random>


#include <unordered_map>
#include <stdexcept>

#include <map>
#include <memory>
#include <functional>
#include <fstream>
#include <sstream>
#include <algorithm>

const std::unordered_set<char> grammar = {
    ',', '.', '!', '?', ';', ':', '"',
    '(', ')', '[', ']', '{', '}', '-',
    '_', '/', '\\', '@', '#', '$', '%',
    '^', '&', '*', '~', '`', '\''
};

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string buffer;

    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!buffer.empty()) {
                tokens.push_back(buffer);
                buffer.clear();
            }
        }
        else if (grammar.count(c)) {
            if (!buffer.empty()) {
                tokens.push_back(buffer);
                buffer.clear();
            }
            tokens.emplace_back(1, c);
        }
        else {
            buffer += c;
        }
    }


    if (!buffer.empty()) {
        tokens.push_back(buffer);
    }

    return tokens;
}

std::string read_file(const std::string &filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file");
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();  // read everything

    return buffer.str();
}


class vocabulary {
    std::unordered_map<std::string, size_t> token_to_id;
    std::vector<std::string> id_to_token;
       public:
    size_t unk_id; // <UNK> token id

    vocabulary() {
        // Reserve special tokens
       // add_token("<PAD>"); // 0
       // add_token("<UNK>"); // 1
        unk_id = 1;
    }

    size_t add_token(const std::string &token) {
        auto it = token_to_id.find(token);
        if (it != token_to_id.end())
            return it->second;

        size_t id = id_to_token.size();
        token_to_id[token] = id;
        id_to_token.push_back(token);
        return id;
    }
    
    size_t token_to_index(const std::string &token) const {
        auto it = token_to_id.find(token);
        if (it != token_to_id.end())
            return it->second;

        return unk_id; // unknown token
    }

    std::string index_to_token(size_t id) const {
        if (id >= id_to_token.size())
            throw std::out_of_range("Token ID out of range");
        return id_to_token[id];
    }

    size_t size() const {
        return id_to_token.size();
    }


    
void feed_vocabulary(int data_size) {
   std::string path = "resources/essay";
  /*
    for(int i = 0 ; i < data_size ; i++) {
         std::string full_path = path + std::to_string(i) + ".txt";
         std::string content = read_file(full_path);
         auto tokens = tokenize(content);
         for( auto token : tokens) {
              //  add_token(token);
         }
    }
*/

    std::string words = "resources/tenta.txt";

    auto newtokens = tokenize(read_file(words));
    for( auto token : newtokens) {
          add_token(token); 
     }
    }


};



struct EmbeddingModel {
   size_t dimension;
   size_t vocab_size;
   
   tensor2 embeddings; // vocab_size x dimension

   EmbeddingModel(size_t vocab_size_, size_t dimension_)
       : vocab_size(vocab_size_), dimension(dimension_), embeddings(vocab_size_, tensor1(dimension_)) {
      
        float limit = std::sqrt(6.0f / (vocab_size + dimension));
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-limit, limit);

        for (size_t i = 0; i < vocab_size; i++)
            for (size_t j = 0; j < dimension; j++)
                embeddings[i][j] = dist(gen);     
   }

    tensor1 get_embedding(size_t index) const {
         if (index >= vocab_size) {
              throw std::out_of_range("Index out of range in get_embedding.");
         }
         return embeddings[index];
    }
};


tensor2 initRandomTensor2(size_t rows, size_t cols,float seed, float lower=-0.1f, float upper=0.1f) {
    tensor2 tensor(rows, tensor1(cols));

    float limit = std::sqrt(6.0f / (rows + cols));
    std::mt19937 gen(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<float> dist(lower, upper);   
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            tensor[i][j] = dist(gen);

    return tensor;
}
