#pragma once

#include "tensor.cc"
#include "Transformer.cc"

#include <algorithm>

#define BCONV 2.0/sqrt(3.141592654)
#define BCONV1 10.0*BCONV
#define BCONV2 100.0*BCONV

// GLEU activation function converges to the ReLU as BCONV approaches infinity 

float GeLU(float x) {
   float erf = std::tanh(BCONV*x);
   
   return 0.5f * x * (1.0f + erf); 
}

float dGeLU(float x) {
   float erf = std::tanh(BCONV*x);
   float derf = BCONV * (1.0f - erf * erf);
   
   return 0.5f * (1.0f + erf) + 0.5f * x * derf; 
}

tensor2 MultiLayerPreceptron(const tensor2 &embeddings,const tensor2 W1,const tensor1 b1,const tensor2 W2,const tensor1 b2) {
   
   if(W2.size() != W1[0].size()) {
       throw std::runtime_error("W2 rows must be equal to W1 columns");
   }

   tensor2 hiddenEmbeddings(embeddings.size(),tensor1(W1[0].size(),0.0f));

   for(int i = 0 ; i < embeddings.size() ; i++) {
       for(int j = 0 ; j < W1[0].size() ; j++) {
           hiddenEmbeddings[i][j] = b1[j];
           for(int k = 0 ; k < W1.size() ; k++) {
               hiddenEmbeddings[i][j] += embeddings[i][k] * W1[k][j];
           }
           hiddenEmbeddings[i][j] = GeLU(hiddenEmbeddings[i][j]);
       }
   }

    tensor2 outputEmbeddings(embeddings.size(),tensor1(W2[0].size(),0.0f));

    for(int i = 0 ; i < hiddenEmbeddings.size() ; i++) {
        for(int j = 0 ; j < W2[0].size() ; j++) {
            outputEmbeddings[i][j] = b2[j];
            for(int k = 0 ; k < W2.size() ; k++) {
                outputEmbeddings[i][j] += hiddenEmbeddings[i][k] * W2[k][j];
            }
        }
    }

    return outputEmbeddings;
}

