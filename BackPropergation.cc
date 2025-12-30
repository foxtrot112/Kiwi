#pragma once

#include "Embedding.cc"
#include "Attention.cc"
#include "FFN.cc"





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
    int dModel,
    int Heads
) {
    AttentionCache cache;


    int dKQV = dModel / Heads;

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
    const tensor2 &dLdO, // upstream gradient from residual/LayerNorm
    const tensor2 &X,
    AttentionCache &cache,
    tensor3 &WQ_heads,
    tensor3 &WK_heads,
    tensor3 &WV_heads,
    tensor2 &WO,
    tensor2 &dX_input,
    float lr,
    int dModel,
    int Heads
) {
    /*
    int dKQV = dModel / Heads;    

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
        




    }*/
    
    size_t n = X.size();

    int dk = dModel/Heads;


    tensor2 dLdC = dLdO * transpose(WO);
    tensor3 dLdHeadi(Heads,tensor2(dLdC.size(),tensor1(dk)));

    for(int d = 0 ; d < Heads ; d++) {
       for(int i = 0 ; i < dLdC.size(); i++) {
         for(int j = 0 ; j < dk ; j++) {
            dLdHeadi[d][i][j] = dLdC[i][j + d*dk];
         }     
       }
    }

    

    tensor3 dLdAi(Heads); //n x n
    tensor3 dLdVi(Heads); //n x dk
    tensor3 dLdZi(Heads); //n x n

    tensor3 dLdQi(Heads); //n x dk
    tensor3 dLdKi(Heads); //n x dk
    
    tensor3 dLdWq(Heads), dLdWk(Heads), dLdWv(Heads); //dmodel x dk
    
    tensor2 dLdX(n,tensor1(dModel,0.0)); // n x dModel

    for(int i = 0 ; i < Heads ; i++) {
        tensor2 V = cache.V_heads[i];
        tensor2 K = cache.K_heads[i];
        tensor2 Q = cache.Q_heads[i];
        
        tensor2 dLdH = dLdHeadi[i];

        dLdAi[i] = dLdH * transpose(V);
        
        tensor2 A = cache.A_heads[i];
        dLdVi[i] = transpose(A) * dLdH;
        
        dLdZi[i] = tensor2(n,tensor1(n));
        tensor2 diagA(n,tensor1(n,0.0));
         

        
        for(int o = 0 ; o < n ; o++) {
           diagA[o][o] = A[o][o];
        }
        
       

        dLdZi[i] = (diagA - (A*transpose(A)))*dLdAi[i];

   

        dLdQi[i] = (1.0/sqrt(float(dk)))*dLdZi[i]*K;
        dLdKi[i] = (1.0/sqrt(float(dk)))*transpose(dLdZi[i])*Q;
        

        dLdWq[i] = transpose(X) * dLdQi[i];
        dLdWk[i] = transpose(X) * dLdKi[i];
        dLdWv[i] = transpose(X) * dLdVi[i];


        dLdX = dLdX + (dLdQi[i] * transpose(WQ_heads[i]));
        dLdX = dLdX + (dLdKi[i] * transpose(WK_heads[i]));
        dLdX = dLdX + (dLdVi[i] * transpose(WV_heads[i]));

    }

   

    // dWO = concatO^T * dO
   
    
    
  
    for(int h = 0 ; h < Heads ; h++) {

      // std::cout << dLdWq[h][0].size() << "\n";

       WQ_heads[h] = WQ_heads[h] - lr*dLdWq[h];
       WK_heads[h] = WK_heads[h] - lr*dLdWk[h];
       WV_heads[h] = WV_heads[h] - lr*dLdWv[h];



    }  
    
    dX_input = dX_input - lr*dLdX;
    //std::cout << "point reached" << "\n";

     tensor2 dLdWO = (transpose(cache.O)* dLdO);
     WO = WO - lr*dLdWO;
}



struct LayerNormCache {
    tensor2 x_hat;   // normalized
    tensor1 mean;    // per token
    tensor1 var;     // per token
};

LayerNormCache layerNormForward(
    const tensor2& X,
    tensor2& Y,
    float eps = 1e-5f
) {
    int T = X.size();
    int D = X[0].size();

    LayerNormCache cache;
    cache.x_hat.resize(T, tensor1(D));
    cache.mean.resize(T);
    cache.var.resize(T);

    Y.resize(T, tensor1(D));

    for(int t = 0; t < T; t++) {
        float mu = 0.0f;
        for(int i = 0; i < D; i++) mu += X[t][i];
        mu /= (float)D;

        float var = 0.0f;
        for(int i = 0; i < D; i++)
            var += (X[t][i] - mu) * (X[t][i] - mu);
        var /= (float)D;

        float inv_std = 1.0f / std::sqrt(var + eps);

        for(int i = 0; i < D; i++) {
            cache.x_hat[t][i] = (X[t][i] - mu) * inv_std;
            Y[t][i] = cache.x_hat[t][i]; // gamma=1, beta=0
        }

        cache.mean[t] = mu;
        cache.var[t]  = var;
    }

    return cache;
}

tensor2 layerNormBackward(
    const tensor2& dY,
    const LayerNormCache& cache,
    float eps = 1e-5f
) {
    int T = dY.size();
    int D = dY[0].size();

    tensor2 dX(T, tensor1(D, 0.0f));

    for(int t = 0; t < T; t++) {
        float inv_std = 1.0f / std::sqrt(cache.var[t] + eps);

        float mean_dY = 0.0f;
        float mean_dY_xhat = 0.0f;

        for(int i = 0; i < D; i++) {
            mean_dY += dY[t][i];
            mean_dY_xhat += dY[t][i] * cache.x_hat[t][i];
        }

        mean_dY /= (float)D;
        mean_dY_xhat /= (float)D;

        for(int i = 0; i < D; i++) {
            dX[t][i] =
                inv_std * (
                    dY[t][i]
                    - mean_dY
                    - cache.x_hat[t][i] * mean_dY_xhat
                );
        }
    }

    return dX;
}

struct MLPCache {
    tensor2 X;       // input to MLP
    tensor2 H1;      // hidden activations after W1 + b1 (before activation)
    tensor2 A1;      // after activation (ReLU)
    tensor2 H2;      // output of W2 + b2 (pre-residual)
    tensor2 output;  // final output after residual addition
};

// Forward MLP with cache recording
MLPCache multiLayerPreceptronCache(
    const tensor2 &X,       // input to the MLP
    const tensor2 &W1,      // dModel x hidden
    const tensor1 &b1,      // hidden
    const tensor2 &W2,      // hidden x dModel
    const tensor1 &b2       // dModel
) {
    MLPCache cache;
    cache.X = X;

    int T = X.size();       // number of tokens
    int D = X[0].size();    // input dimension (dModel)
    int H = W1[0].size();   // hidden dimension

    cache.H1.resize(T, tensor1(H, 0.0f));
    cache.A1.resize(T, tensor1(H, 0.0f));

    // --- First Linear Layer + ReLU ---
    for(int t = 0; t < T; t++) {
        for(int h = 0; h < H; h++) {
            float sum = b1[h];
            for(int d = 0; d < D; d++)
                sum += X[t][d] * W1[d][h];
            cache.H1[t][h] = sum;
            cache.A1[t][h] = std::max(0.0f, sum); // ReLU
        }
    }

    // --- Second Linear Layer ---
    cache.H2.resize(T, tensor1(D, 0.0f));
    cache.output.resize(T, tensor1(D, 0.0f));
    for(int t = 0; t < T; t++) {
        for(int d = 0; d < D; d++) {
            float sum = b2[d];
            for(int h = 0; h < H; h++)
                sum += cache.A1[t][h] * W2[h][d];
            cache.H2[t][d] = sum;
            cache.output[t][d] = sum + X[t][d]; // residual connection
        }
    }

    return cache;
}
void backpropMLP(
    const MLPCache &cache,
    tensor2 &W1, tensor1 &b1,
    tensor2 &W2, tensor1 &b2,
    const tensor2 &dOutput,   // upstream gradient (same shape as cache.output)
    tensor2 &dX,              // gradient w.r.t input X
    float lr                  // learning rate
) {
    int T = cache.X.size();    // number of tokens
    int D = cache.X[0].size(); // input dimension
    int H = W1[0].size();      // hidden dimension

    // Gradients for W2, b2, A1
    tensor2 dW2(H, tensor1(D, 0.0f));
    tensor1 db2(D, 0.0f);
    tensor2 dA1(T, tensor1(H, 0.0f));

    // --- Backprop through second linear + residual ---
    dX.resize(T, tensor1(D, 0.0f));
    for(int t = 0; t < T; t++) {
        for(int d = 0; d < D; d++) {
            float grad = dOutput[t][d]; // upstream gradient
            db2[d] += grad;
            for(int h = 0; h < H; h++) {
                dW2[h][d] += cache.A1[t][h] * grad;
                dA1[t][h] += W2[h][d] * grad;
            }
            dX[t][d] += grad; // residual connection
        }
    }

    // --- Backprop through ReLU ---
    tensor2 dH1(T, tensor1(H, 0.0f));
    for(int t = 0; t < T; t++) {
        for(int h = 0; h < H; h++) {
            dH1[t][h] = (cache.H1[t][h] > 0.0f) ? dA1[t][h] : 0.0f;
        }
    }

    // --- Backprop through first linear layer ---
    tensor2 dW1(D, tensor1(H, 0.0f));
    tensor1 db1(H, 0.0f);

    for(int t = 0; t < T; t++) {
        for(int h = 0; h < H; h++) {
            db1[h] += dH1[t][h];
            for(int d = 0; d < D; d++) {
                dW1[d][h] += cache.X[t][d] * dH1[t][h];
                dX[t][d] += W1[d][h] * dH1[t][h];
            }
        }
    }

    // --- Update weights and biases ---
    for(int d = 0; d < D; d++)
        for(int h = 0; h < H; h++)
            W1[d][h] -= lr * dW1[d][h];

    for(int h = 0; h < H; h++)
        b1[h] -= lr * db1[h];

    for(int h = 0; h < H; h++)
        for(int d = 0; d < D; d++)
            W2[h][d] -= lr * dW2[h][d];

    for(int d = 0; d < D; d++)
        b2[d] -= lr * db2[d];
}