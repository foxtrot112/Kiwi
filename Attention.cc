#pragma once
#include "tensor.cc"
#include "Transformer.cc"


tensor3 linearOKV(const tensor2 &embedding_input,const tensor3 &QKV_weights,const tensor2 &QKV_biases) {
   auto Wq = QKV_weights[0];
   auto Wk = QKV_weights[1];
   auto Wv = QKV_weights[2];

   auto bq = QKV_biases[0];
   auto bk = QKV_biases[1];
   auto bv = QKV_biases[2];

   
   tensor2 Q(embedding_input.size(),tensor1(Wq.size())),K(embedding_input.size(),tensor1(Wk.size())),V(embedding_input.size(),tensor1(Wv.size()));
   
   for(int i = 0 ; i < embedding_input.size(); i++) {
       for(int j = 0; j < Wq.size(); j++) {
           Q[i][j] = bq[j];
           for(int k = 0; k < embedding_input[0].size(); k++) {
               Q[i][j] += embedding_input[i][k] * Wq[k][j];
           }
       }
       for(int j = 0; j < Wk.size(); j++) {
           K[i][j] = bk[j];
           for(int k = 0; k < embedding_input[0].size(); k++) {
               K[i][j] += embedding_input[i][k] * Wk[k][j];
           }
       }
       for(int j = 0; j < Wv.size(); j++) {
           V[i][j] = bv[j];
           for(int k = 0; k < embedding_input[0].size(); k++) {
               V[i][j] += embedding_input[i][k] * Wv[k][j];
           }
       }
   }

  

   return {Q,K,V};
}

tensor2 singleHeadAttention(const tensor2 &Q,const tensor2 &K,const tensor2 &V,const int dModel) {
    tensor2 scores(Q.size(),tensor1(K.size()));
    
    for(int u = 0 ; u < Q.size() ; u++) {
      for(int v = 0 ; v < K.size() ; v++) {
         scores[u][v] = dot(Q[u],K[v]) / sqrt((float)dModel); 
      }  
    }
    
    tensor2 attentionWeights = softMax(scores);
     
   return attentionWeights * V;
}


tensor2 multiHeadAttention(const tensor2 &embedding_input,const tensor3 &WQi,const tensor3 &Wki,const tensor3 &Wvi,const tensor2 &Wo,const float dModel,const int numHeads) {
    int headDim = dModel / numHeads;
    if(dModel != headDim * numHeads) {
        throw std::runtime_error("dModel must be equal to headDim * numHeads");
    }
    
    tensor2 multiHeadOutput(embedding_input.size(),tensor1(dModel,0.0f));
    
    tensor3 headOutputs(numHeads);

    for(int i = 0 ; i < numHeads ; i++) {
        auto Wq = WQi[i];
        auto Wk = Wki[i];
        auto Wv = Wvi[i];

        tensor2 Qi(embedding_input.size(),tensor1(headDim,0.0f));
        tensor2 Ki(embedding_input.size(),tensor1(headDim,0.0f));
        tensor2 Vi(embedding_input.size(),tensor1(headDim,0.0f));

        for(int u = 0 ; u < embedding_input.size() ; u++) {
            for(int j = 0 ; j < headDim ; j++) {
                for(int k = 0 ; k < dModel ; k++) {
                    Qi[u][j] += embedding_input[u][k] * Wq[k][j];
                    Ki[u][j] += embedding_input[u][k] * Wk[k][j];
                    Vi[u][j] += embedding_input[u][k] * Wv[k][j];
                }
            }
        }


        headOutputs[i] = singleHeadAttention(Qi,Ki,Vi,(int)headDim);

    }

    for(int i = 0 ; i < embedding_input.size() ; i++) {
        for(int h = 0 ; h < numHeads ; h++) {
            for(int j = 0 ; j < headDim ; j++) {
                multiHeadOutput[i][h * headDim + j] = headOutputs[h][i][j];
            }
        }
    }

    for(int i = 0 ; i < embedding_input.size() ; i++) {
        for(int j = 0 ; j < dModel ; j++) {
            float sum = 0.0f;
            for(int k = 0 ; k < dModel ; k++) {
                sum += multiHeadOutput[i][k] * Wo[k][j];
            }
            multiHeadOutput[i][j] = sum;
        }
    }



    return multiHeadOutput;
}




