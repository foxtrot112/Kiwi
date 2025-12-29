#pragma once

#include "Embedding.cc"
#include "Attention.cc"
#include "FFN.cc"

#define dModel 512
#define Heads 8
#define dKQV (dModel/Heads)

#define MLP_hidden (dModel*4)
#define Nx 4 //layer number
///Total tunnable parameter without any normalization or quantization is given by (heads*3 + 1 + 2)*Nx : weights matrices , bais = 3*Nx

#define searchWords 10 
#define smoothingHot 5


void propergate_projection_weight_extended(tensor2 &output_projection_weight,tensor1 embeddingVector,
                                                                    tensor1 current_probVector,
                                                                    tensor1 actual_one_hot,float lr) {
    tensor3 dLi_dWo(current_probVector.size(),tensor2(embeddingVector.size(),tensor1(current_probVector.size(),0.0f)));
    
    for(int i = 0 ; i < current_probVector.size() ; i++) {
        for(int j = 0 ; j < embeddingVector.size() ; j++) {
           for(int k = 0 ; k < current_probVector.size() ; k++){
                if(i != j) {
                    dLi_dWo[i][j][k] = 0.0f;
                } 
                else {
                    float sum = 0.0;

                    for(int d = 0 ; d < j; d++) {
                        sum += embeddingVector[d];
                    }

                    dLi_dWo[i][j][k] = sum;
                           
                }
            
        }
      }
    }

    tensor2 dSoftmax_dLi(current_probVector.size(),tensor1(current_probVector.size(),0.0f));

    for(int i = 0 ; i < current_probVector.size() ; i++) {
        for(int j = 0 ; j < current_probVector.size() ; j++) {
            if(i == j) {
                dSoftmax_dLi[i][j] = current_probVector[i] * (1 - current_probVector[i]);
            } else {
                dSoftmax_dLi[i][j] = -current_probVector[i] * current_probVector[j];
            }
        }
    }


    tensor3 dprob_dWo(current_probVector.size(),tensor2(embeddingVector.size(),tensor1(current_probVector.size(),0.0f)));

    for(int i = 0 ; i < current_probVector.size() ; i++) {
        for(int j = 0 ; j < embeddingVector.size() ; j++) {
            for(int k = 0 ; k < current_probVector.size() ; k++) {
                float sum = 0.0f;
                for(int l = 0 ; l < current_probVector.size() ; l++) {
                    sum += dSoftmax_dLi[i][l] * dLi_dWo[l][j][k];
                }
                dprob_dWo[i][j][k] = sum;
            }
        }
    } 
     
    tensor3 dprobTila_dWo(current_probVector.size(),tensor2(embeddingVector.size(),tensor1(current_probVector.size(),0.0f)));
    

 
    for(int i = 0 ; i < current_probVector.size() ; i++) {
        for(int j = 0 ; j < embeddingVector.size() ; j++) {
            for(int k = 0 ; k < current_probVector.size() ; k++) { 
                dprobTila_dWo[i][j][k] = 1.0/current_probVector[i] * dprob_dWo[i][j][k];
            }
        }
    }

    tensor2 dL_dWo(embeddingVector.size(),tensor1(current_probVector.size(),0.0f));


    for(int i = 0 ; i < embeddingVector.size() ; i++) {
        for(int j = 0 ; j < current_probVector.size() ; j++) {
            float sum = 0.0f;
            for(int k = 0 ; k < current_probVector.size() ; k++) {
                sum += -actual_one_hot[k] * dprobTila_dWo[k][i][j];
            }
            dL_dWo[i][j] = sum;
        }
    }




   

    output_projection_weight = output_projection_weight - lr*dL_dWo;
 }
                                                                      

 void propergate_projection_weight(tensor2 &output_projection_weight,tensor1 embeddingVector,
                                                                    tensor1 current_probVector,
                                                                    tensor1 actual_one_hot,float lr) {
  
    tensor2 dL_dWo(embeddingVector.size(),tensor1(current_probVector.size(),0.0f));


    for(int i = 0 ; i < embeddingVector.size() ; i++) {
        for(int j = 0 ; j < current_probVector.size() ; j++) {
            dL_dWo[i][j] = (current_probVector[j] - actual_one_hot[j]) * embeddingVector[i];
        }
    }

   output_projection_weight = output_projection_weight - lr*dL_dWo;
}



 void propergate_projection_bais(tensor1 &output_projection_bais,
                                                                    tensor1 current_probVector,
                                                                    tensor1 actual_one_hot,float lr) {

    tensor1 dL_db(current_probVector.size(),0.0f);

    for(int j = 0 ; j < current_probVector.size() ; j++) {
        dL_db[j] = current_probVector[j] - actual_one_hot[j];
    }

    output_projection_bais = output_projection_bais - lr*dL_db;

                                                                    }







struct AttentionCache {
    std::vector<tensor2> Q_heads;
    std::vector<tensor2> K_heads;
    std::vector<tensor2> V_heads;
    std::vector<tensor2> A_heads;
    tensor2 O; // final output after WO
};

AttentionCache multiHeadAttentionCache(
    const tensor2 &X,
    const tensor3 &WQ_heads,
    const tensor3 &WK_heads,
    const tensor3 &WV_heads,
    const tensor2 &WO,
    int _dModel,
    int _Heads
) {
    AttentionCache cache;
    int _dKQV = dModel / Heads;

    cache.Q_heads.resize(Heads);
    cache.K_heads.resize(Heads);
    cache.V_heads.resize(Heads);
    cache.A_heads.resize(Heads);

    tensor2 concatO(X.size(), tensor1(dModel,0.0f)); // for concatenated heads

    for(int h = 0 ; h < Heads ; h++){
        // Linear projections
        cache.Q_heads[h] = X * WQ_heads[h];
        cache.K_heads[h] = (X * WK_heads[h]);
        cache.V_heads[h] = (X  * WV_heads[h]);

        // Attention scores
        tensor2 S = (cache.Q_heads[h] * transpose(cache.K_heads[h]));
        for(int i = 0 ; i < S.size(); i++)
            for(int j = 0 ; j < S[0].size(); j++)
                S[i][j] /= sqrtf((float)dKQV);

        // Softmax
        cache.A_heads[h] = softMax(S);

        // Output of this head
        tensor2 O_h = (cache.A_heads[h] * cache.V_heads[h]);

        // Concatenate into concatO
        for(int i = 0 ; i < X.size(); i++)
            for(int j = 0 ; j < dKQV; j++)
                concatO[i][h*dKQV + j] = O_h[i][j];
    }

    // Final projection
    cache.O = (concatO * WO);

    return cache;
}


void backpropMultiHeadAttention(
    const tensor2 &dO, // upstream gradient from residual/LayerNorm
    const tensor2 &X,
    AttentionCache &cache,
    tensor3 &WQ_heads,
    tensor3 &WK_heads,
    tensor3 &WV_heads,
    tensor2 &WO,
    tensor2 &dX_input,
    float lr
) {


    // 1. Grad WO
    tensor2 concatO(X.size(), tensor1(dModel,0.0f));
    for(int i = 0 ; i < X.size(); i++)
        for(int h = 0 ; h < Heads ; h++)
            for(int j = 0 ; j < dKQV ; j++)
                concatO[i][h*dKQV + j] = cache.A_heads[h][i][j]; // or O_h?

    // dWO = concatO^T * dO
    tensor2 dWO = (transpose(concatO)* dO);

    // Update WO
    for(int i = 0 ; i < WO.size(); i++)
        for(int j = 0 ; j < WO[0].size(); j++)
            WO[i][j] -= lr * dWO[i][j];

    // 2. Grad w.r.t concatO
    tensor2 dConcatO = (dO * transpose(WO));

    // 3. Split per head and backprop A and V
    dX_input = tensor2(X.size(), tensor1(dModel,0.0f));

    for(int h = 0 ; h < Heads ; h++){
        // Extract dO_h
        tensor2 dO_h(X.size(), tensor1(dKQV,0.0f));
        for(int i = 0 ; i < X.size(); i++)
            for(int j = 0 ; j < dKQV; j++)
                dO_h[i][j] = dConcatO[i][h*dKQV + j];

        // Grad V
        tensor2 dV = (transpose(cache.A_heads[h]) * dO_h);

        // Update WV
        tensor2 dWV = (transpose(X) * dV);
        for(int i = 0 ; i < WV_heads[h].size(); i++)
            for(int j = 0 ; j < WV_heads[h][0].size(); j++)
                WV_heads[h][i][j] -= lr * dWV[i][j];

        // Grad w.r.t A
        tensor2 dA = (dO_h * transpose(cache.V_heads[h]));

        // Grad w.r.t S (softmax)
        tensor2 dS = tensor2(dA.size(), tensor1(dA[0].size(),0.0f));
        for(int i = 0 ; i < dA.size(); i++){
            for(int j = 0 ; j < dA[0].size(); j++){
                float sum = 0.0f;
                for(int k = 0 ; k < dA[0].size(); k++){
                    float delta = (j==k?1.0f:0.0f);
                    sum += dA[i][k] * cache.A_heads[h][i][k] * (delta - cache.A_heads[h][i][j]);
                }
                dS[i][j] = sum;
            }
        }

        // Grad w.r.t Q and K
        tensor2 dQ = (dS * cache.K_heads[h]) * (1.0f / sqrtf((float)dKQV));
        tensor2 dK = (transpose(dS) * cache.Q_heads[h]) * (1.0f / sqrtf((float)dKQV));

        // Update WQ and WK
        tensor2 dWQ = (transpose(X) * dQ);
        tensor2 dWK = (transpose(X) * dK);

        for(int i = 0 ; i < WQ_heads[h].size(); i++)
            for(int j = 0 ; j < WQ_heads[h][0].size(); j++)
                WQ_heads[h][i][j] -= lr * dWQ[i][j];

        for(int i = 0 ; i < WK_heads[h].size(); i++)
            for(int j = 0 ; j < WK_heads[h][0].size(); j++)
                WK_heads[h][i][j] -= lr * dWK[i][j];

        // Grad w.r.t input X from this head
        dX_input = (dX_input + (dQ * transpose(WQ_heads[h])) + (dK * transpose(WK_heads[h])) + (dV * transpose(WV_heads[h])));
    }
}
