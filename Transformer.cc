#pragma once

#include "tensor.cc"

tensor2 softMax(const tensor2 &scores) {
    tensor2 softmaxed(scores.size(),tensor1(scores[0].size()));
    
    for(int i = 0 ; i < scores.size() ; i++) {
        float maxScore = *std::max_element(scores[i].begin(),scores[i].end());
        float sumExp = 0.0f;
        for(int j = 0 ; j < scores[i].size() ; j++) {
            sumExp += std::exp(scores[i][j] - maxScore);
        }
        for(int j = 0 ; j < scores[i].size() ; j++) {
            softmaxed[i][j] = std::exp(scores[i][j] - maxScore) / sumExp;
        }
    }
    
    return softmaxed ;
}

tensor2 layerNorm(const tensor2 &X,const float gamma,const float beta) {
   tensor2 normalizedX(X.size(),tensor1(X[0].size(),0.0f));

   for(int i = 0 ; i < X.size() ; i++) {
       float mean = 0.0f;
       for(int j = 0 ; j < X[0].size() ; j++) {
           mean += X[i][j];
       }
       mean /= X[0].size();

       float variance = 0.0f;
       for(int j = 0 ; j < X[0].size() ; j++) {
           variance += (X[i][j] - mean) * (X[i][j] - mean);
       }
       variance /= X[0].size();

       for(int j = 0 ; j < X[0].size() ; j++) {
           normalizedX[i][j] = gamma * (X[i][j] - mean) / std::sqrt(variance + 1e-5f) + beta;
       }
   }
  
    return normalizedX;
}


tensor3 partial_concate(const tensor3 &tensors) {
   size_t N = tensors.size();
   size_t dk = tensors[0][0].size();

   size_t seqLen = tensors[0].size();

   
  
   tensor3 partial_concated;

    for(size_t i = 0 ; i < N ; i++) {
       tensor2 concated(seqLen,tensor1(N * dk,0.0f));
         for(size_t j = 0 ; j < seqLen ; j++) {
              for(size_t k = 0 ; k < dk ; k++) {
                concated[j][i * dk + k] = tensors[i][j][k];
              }
         }

        partial_concated.push_back(concated);
    }
  return partial_concated;
   
  //the idea of this is to concentrate on the concatenation along the feature dimension only
  // in partial derivates some aera in the concated matrix is not dependednt on the corresponding area in the input tensor
  // so evenutally some partial derivatives with the respect of those areas will be zero
  /*
  |0 0 0 0 0 a b c d e 0 0 0 0 0|
  |0 0 0 0 0 f g h i j 0 0 0 0 0|
  |0 0 0 0 0 k l m n o 0 0 0 0 0| 
  |0 0 0 0 0 p q r s t 0 0 0 0 0|
  |0 0 0 0 0 u v w x y 0 0 0 0 0|

  */

}

tensor2 concate(const tensor3 &tensors) {
   size_t N = tensors.size();
   size_t dk = tensors[0][0].size();

   size_t seqLen = tensors[0].size();

   tensor2 concated(seqLen,tensor1(N * dk,0.0f));
   tensor3 partial_concated = partial_concate(tensors);

    for(size_t i = 0 ; i < N ; i++) {
      concated = concated + partial_concated[i];
    }

    return concated;
}