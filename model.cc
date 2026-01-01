#pragma once 


#include "Embedding.cc"
#include "Attention.cc"
#include "FFN.cc"
#include "BackPropergation.cc"

#include "examples.cc"

#include "TunningCache.cc"
#include <iostream>
#include <chrono>
#include <thread>


struct LayerPeramaters {
   tensor3 pWQ_heads; //Heads x dModel x dKQV
   tensor3 pWK_heads; //Heads x dModel x dKQV
   tensor3 pWV_heads; //Heads x dModel x dKQV

   tensor2 WO ; // dModel x dModel
   tensor1 bO ; // dModel

   tensor2 W1 ; // dModel x MLP_hidden
   tensor1 b1 ; // MLP_hidden

   tensor2 W2 ; // MLP_hidden x dModel
   tensor1 b2 ; // dModel
};

struct labels {
    std::string hot_label;
    std::string training_label;
};

struct LayerCacheConfig {
   std::vector<std::string> WQ_cache_path;
   std::vector<std::string> WK_cache_path;
   std::vector<std::string> WV_cache_path;
   std::string Wo_cache_path;

   std::string W1_cache_path;
   std::string W2_cache_path;
   std::string b1_cache_path;
   std::string b2_cache_path;
};

struct projectionCacheConfig {
  std::string Wprojection_cache_path;
  std::string b_projection_cache_path;
};


class Model {
  public:


   vocabulary vocab;


    size_t _dModel;
    size_t _HeadN;
    size_t _LayerN;

    size_t _epoch;
    float learningRate;
     std::vector<LayerPeramaters> layers_params;

    tensor2 output_projection_weights;
    tensor1 output_projection_bias;

    std::vector<std::string> training_examples;
    
    std::string layerCachePaths;
    std::string projectionCachePaths;

   
   void train(EmbeddingModel &model,size_t overide_example_training,size_t token_intial);
   void generates(EmbeddingModel &model,std::string intial_prompt,size_t number_of_tokens,std::string &output);


   private:
     
     labels getLabels(std::string example,size_t token_intial);  

     void saveCache();
};


labels Model::getLabels(std::string example,size_t token_intial) {
      std::string training_label = example;
      std::vector<std::string> label_tokens = tokenize(training_label);

      size_t char_initials = 0;
      for(int qq = 0; qq < token_intial && qq < label_tokens.size(); qq++) {
          char_initials += label_tokens[qq].size();
          if (qq < token_intial - 1) char_initials += 1; // account for space
       }
     
      std::string initial_label = training_label.substr(0, char_initials);
      std::string hot_label = training_label.substr(char_initials);

    return {hot_label,initial_label};  
}

size_t sample_from_distribution(const tensor1& probs) {
    float r = static_cast<float>(rand()) / RAND_MAX; // uniform random [0,1)
    float accum = 0.0f;
    
    for(size_t i = 0; i < probs.size(); ++i) {
        accum += probs[i];
        if (r < accum)
            return i;
    }
    
    return probs.size() - 1; // fallback
}

void Model::train(EmbeddingModel& model,size_t overide_example_training,size_t token_intial) {
   if(overide_example_training > training_examples.size()) 
     throw std::runtime_error("excessive amout of training request.");
   
   for(int e = 0 ; e < _epoch ; e++) {
     std::cout << "epochLevel:" << " " << e << "\n";

     for(int d = 0; d < overide_example_training ; d++) {
       labels example_labels = getLabels(training_examples[d],token_intial);
       
       
       std::vector<std::string> hot_tokens = tokenize(example_labels.hot_label);
       
       
       std::cout << "\r" << example_labels.training_label << std::flush;
       std::vector<float> token_losses;
       for(int t = 0 ; t < hot_tokens.size() ; t++) {

       std::vector<std::string> tokens = tokenize(example_labels.training_label);
         
        
         tensor2 embedding_input;

          for(auto w:tokens) {
            size_t token_id = vocab.token_to_index(w);
            embedding_input.push_back(model.get_embedding(token_id));
          }

         embedding_input = embedding_input + create_position_encoding(tokens.size(),_dModel);  
        
         std::vector<tensor2> layer_inputs(_LayerN);
         std::vector<AttentionCache> Attention_caches(_LayerN);
         std::vector<LayerNormCache> LayerNorm1_caches(_LayerN); // LayerNorm after attention
         std::vector<MLPCache> MLP_caches(_LayerN);            // you need to define this struct
         std::vector<LayerNormCache> LayerNorm2_caches(_LayerN); // LayerNorm after MLP
         
         
  for(int n = 0 ; n < _LayerN ; n++) {
    layer_inputs[n] = embedding_input; // input to attention + residual

    // --- Multi-Head Attention ---
    Attention_caches[n] = multiHeadAttentionCache(
        embedding_input,
        layers_params[n].pWQ_heads,
        layers_params[n].pWK_heads,
        layers_params[n].pWV_heads,
        layers_params[n].WO,
        _dModel,
        _HeadN
    );

    tensor2 attention_output = Attention_caches[n].O;

    // --- LayerNorm after Attention (first LN) ---
    tensor2 ln1_output;
    LayerNorm1_caches[n] = layerNormForward(attention_output + embedding_input, ln1_output);
    embedding_input = ln1_output;

    // --- MLP ---
    MLP_caches[n] = multiLayerPreceptronCache(embedding_input, layers_params[n].W1, layers_params[n].b1,
                                               layers_params[n].W2, layers_params[n].b2); // store input + pre-activations
    tensor2 mlp_output = MLP_caches[n].output;

    // --- LayerNorm after MLP (second LN) ---
    tensor2 ln2_output;
    LayerNorm2_caches[n] = layerNormForward(mlp_output + embedding_input, ln2_output);
    embedding_input = ln2_output; // this is the new input to the next layer
}

         tensor1 targeted_embedding = embedding_input[tokens.size() - 1];
        
         tensor1 logits(vocab.size(),0.0);
         
         for(int i = 0 ; i < vocab.size() ; i++) {
           for(int u = 0 ; u < targeted_embedding.size() ; u++) {
             logits[i] += targeted_embedding[u] * output_projection_weights[u][i] + output_projection_bias[i] + 1e-4;
           }
         }

         size_t testing_nan = std::distance(logits.begin(), std::max_element(logits.begin(),logits.end()));

        // std::cout << logits[testing_nan] << "\n";

         tensor1 probabilities = softMax({logits})[0];
        // probabilities[vocab.token_to_index("<PAD>")] = 0.0f;
         
         


        size_t predicted_index = std::distance(probabilities.begin(), std::max_element(probabilities.begin(),probabilities.end()));
         
       float temperature = 0.8f;

    //for (float &p : probabilities)
   // p = std::pow(p, 1.0f / temperature);

// renormalize
       //float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
         //for (float &p : probabilities)
         //p /= sum;

        // size_t predicted_index = sample_from_distribution(probabilities);
        // std::cout << probabilities[predicted_index] << "\n";


         std::string the_actual_word = hot_tokens[t];
         size_t actual_index = vocab.token_to_index(the_actual_word);

         tensor1 hot_probabilities(probabilities.size(),0.0);
         
         hot_probabilities[actual_index] = 1.0;

         float eps_label_symbol = 0.1;

         for(auto &y:hot_probabilities)
           y = (1.0 - eps_label_symbol)*y + eps_label_symbol/vocab.size();
          
          float cross_entropy_loss = 0.0f;

          for(int i = 0 ; i < hot_probabilities.size() ; i++) {
            cross_entropy_loss += hot_probabilities[i] * std::log(probabilities[i] + 1e-5);
          }

          cross_entropy_loss *= -1.0f;
          token_losses.push_back(cross_entropy_loss);
  
        example_labels.training_label += " " + vocab.index_to_token(predicted_index);
       std::cout <<"\r"  <<  " " + vocab.index_to_token(predicted_index) << std::flush ; 


        // if (the_actual_word == "<PAD>") continue;

         /// Backpropergation :
        tensor1 dTargetedEmbedding(_dModel, 0.0f);

        for(int k = 0; k < _dModel; k++) {
          for(int j = 0; j < vocab.size(); j++) {
            dTargetedEmbedding[k] +=
            output_projection_weights[k][j] *
            (probabilities[j] - hot_probabilities[j]);
            }
        } 

      tensor2 dEmbeddingInput(tokens.size(), tensor1(_dModel, 0.0f));

      // Only last token receives gradient
      dEmbeddingInput[tokens.size() - 1] = dTargetedEmbedding;  


      // Backprop through layers
       
tensor2 dLdO = dEmbeddingInput;


//tensor2 dX = dEmbeddingInput; // from output projection

for(int bn = _LayerN - 1; bn >= 0; --bn) {
    // 1) LayerNorm after MLP
    tensor2 dX_ln2 = layerNormBackward(dLdO, LayerNorm2_caches[bn]);

    // 2) Backprop through MLP
    tensor2 dX_mlp;
    backpropMLP(MLP_caches[bn], 
                layers_params[bn].W1, layers_params[bn].b1,
                layers_params[bn].W2, layers_params[bn].b2,
                dX_ln2, dX_mlp, learningRate);


    tensor2 dX_ln1 = layerNormBackward(dX_mlp + dX_ln2, LayerNorm1_caches[bn]);

    tensor2 dX_residual = dX_ln1;

    // 3) Backprop through Attention + LayerNorm
    tensor2 dX_attention(dX_residual.size(),tensor1(dX_residual[0].size(),0.0f));
    backpropMultiHeadAttention(dX_residual, layer_inputs[bn],
                               Attention_caches[bn],
                               layers_params[bn].pWQ_heads,
                               layers_params[bn].pWK_heads,
                               layers_params[bn].pWV_heads,
                               layers_params[bn].WO,
                               dX_attention, learningRate, _dModel, _HeadN);

    // 4) Combine residual
    dLdO = dX_residual + dX_attention;
}



       //Backprop Through the projection peramaters


       propergate_projection_weight(output_projection_weights, targeted_embedding, probabilities, hot_probabilities, learningRate);
       propergate_projection_bais(output_projection_bias, probabilities, hot_probabilities, learningRate);
         
       saveCache();


       
       }
       
       float average = 0.0;
         for(auto loss:token_losses) 
           average += loss;
       
       std::cout << "\n";
       std::cout << average / token_losses.size() << "\n";
       
     }

     std::cout << "\n";
   }  
 }



void Model::generates(EmbeddingModel &model,std::string intial_prompt,size_t number_of_tokens,std::string &output) {
     
    std::string prompt = intial_prompt;
    

    for(int t = 0 ; t < number_of_tokens ; t++) {
        std::vector<std::string> tokens = tokenize(prompt);

         tensor2 embedding_input;

          for(auto w:tokens) {
            size_t token_id = vocab.token_to_index(w);
            embedding_input.push_back(model.get_embedding(token_id));
          }
        
               
         std::vector<tensor2> layer_inputs(_LayerN);
         std::vector<AttentionCache> Attention_caches(_LayerN);
         std::vector<LayerNormCache> LayerNorm1_caches(_LayerN); // LayerNorm after attention
         std::vector<MLPCache> MLP_caches(_LayerN);            // you need to define this struct
         std::vector<LayerNormCache> LayerNorm2_caches(_LayerN); // LayerNorm after MLP
         
         
           for(int n = 0 ; n < _LayerN ; n++) {
             layer_inputs[n] = embedding_input; // input to attention + residual

             // --- Multi-Head Attention ---
             Attention_caches[n] = multiHeadAttentionCache(
                 embedding_input,
                 layers_params[n].pWQ_heads,
                 layers_params[n].pWK_heads,
                 layers_params[n].pWV_heads,
                 layers_params[n].WO,
                 _dModel,
                 _HeadN
             );

             tensor2 attention_output = Attention_caches[n].O;

             // --- LayerNorm after Attention (first LN) ---
             tensor2 ln1_output;
             LayerNorm1_caches[n] = layerNormForward(attention_output + embedding_input, ln1_output);
             embedding_input = ln1_output;

             // --- MLP ---
             MLP_caches[n] = multiLayerPreceptronCache(embedding_input, layers_params[n].W1, layers_params[n].b1,
                                                        layers_params[n].W2, layers_params[n].b2); // store input + pre-activations
             tensor2 mlp_output = MLP_caches[n].output;

             // --- LayerNorm after MLP (second LN) ---
             tensor2 ln2_output;
             LayerNorm2_caches[n] = layerNormForward(mlp_output + embedding_input, ln2_output);
             embedding_input = ln2_output; // this is the new input to the next layer
        }

         tensor1 targeted_embedding = embedding_input[tokens.size() - 1];

         tensor1 logits(vocab.size(),0.0);
         
         for(int i = 0 ; i < vocab.size() ; i++) {
           for(int u = 0 ; u < targeted_embedding.size() ; u++) {
             logits[i] += targeted_embedding[u] * output_projection_weights[u][i] + output_projection_bias[i];
           }
         }

         tensor1 probabilities = softMax({logits})[0];

    float temperature = 0.7f;

    for (float &p : probabilities)
    p = std::pow(p, 1.0f / temperature);

// renormalize
       float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
       for (float &p : probabilities)
         p /= sum;
         
         size_t predicted_index = sample_from_distribution(probabilities);
        // size_t predicted_index = std::distance(probabilities.begin(), std::max_element(probabilities.begin(),probabilities.end()));
         
         std::cout << probabilities[predicted_index] <<"\n";

         std::string predicted_next_token = vocab.index_to_token(predicted_index);
         
         prompt += " " + predicted_next_token;
         
    }
    
    
    output = prompt;
      
}


void Model::saveCache() {

  std::string extention = ".bin";
  ///Layers
   
  std::string base_Wq = "cacheWQ_";
  std::string base_Wk = "cacheWK_";
  std::string base_Wv = "cacheWV_";
  std::string base_Wo = "cacheWO_";
  
  std::string base_W1 = "cacheW1_";
  std::string base_W2 = "cacheW2_";
  std::string base_b1 = "cacheB1_";
  std::string base_b2 = "cacheB2_";




  for(int n = 0 ; n < _LayerN; n++) {
    for(int h = 0 ; h < _HeadN ; h++) {
       std::string filenameWqni = base_Wq + "n" + std::to_string(n) + "h" + std::to_string(h);
       std::string filenameWkni = base_Wk + "n" + std::to_string(n) + "h" + std::to_string(h);
       std::string filenameWvni = base_Wv + "n" + std::to_string(n) + "h" + std::to_string(h);
       
       saveTensor2Bin(layers_params[n].pWQ_heads[h],layerCachePaths + "/" + filenameWqni + extention);
       saveTensor2Bin(layers_params[n].pWK_heads[h],layerCachePaths + "/" + filenameWkni + extention);
       saveTensor2Bin(layers_params[n].pWV_heads[h],layerCachePaths + "/" + filenameWvni + extention);
    } 
     
       std::string filenameWon = base_Wo + "n" + std::to_string(n);
       std::string filenameW1n = base_W1 + "n" + std::to_string(n);
       std::string filenameW2n = base_W2 + "n" + std::to_string(n);

       std::string filenameB1n = base_b1 + "n" + std::to_string(n);
       std::string filenameB2n = base_b2 + "n" + std::to_string(n);

       saveTensor2Bin(layers_params[n].WO,layerCachePaths + "/" + filenameWon + extention);
       saveTensor2Bin(layers_params[n].W1,layerCachePaths + "/" + filenameW1n + extention); 
       saveTensor2Bin(layers_params[n].W2,layerCachePaths + "/" + filenameW2n + extention);       
       
       saveTensor1Bin(layers_params[n].b1,layerCachePaths + "/" + filenameB1n + extention);
       saveTensor1Bin(layers_params[n].b2,layerCachePaths + "/" + filenameB2n + extention);

  }

  std::string base_Wp = "cacheWP_";
  std::string base_bp = "cacheBP_";
  
  std::string filenameWp = base_Wp + "0";
  std::string filenameBp = base_bp + "0";

  saveTensor2Bin(output_projection_weights,projectionCachePaths + "/" + filenameWp + extention);
  saveTensor1Bin(output_projection_bias,projectionCachePaths + "/" + filenameBp + extention);
 
}


void loadCache(std::vector<LayerPeramaters> &LayerPerams,tensor2 &WeightProjection,tensor1 &biasProjection,size_t _LayerN, size_t _HeadN, std::string layerCachePaths, std::string projectionCachePaths) {
   // std::vector<LayerPeramaters> LayerPeramsOut(_LayerN);
    
      std::string extention = ".bin";
  ///Layers
   
  std::string base_Wq = "cacheWQ_";
  std::string base_Wk = "cacheWK_";
  std::string base_Wv = "cacheWV_";
  std::string base_Wo = "cacheWO_";
  
  std::string base_W1 = "cacheW1_";
  std::string base_W2 = "cacheW2_";
  std::string base_b1 = "cacheB1_";
  std::string base_b2 = "cacheB2_";




  for(int n = 0 ; n < _LayerN; n++) {
    for(int h = 0 ; h < _HeadN ; h++) {
       std::string filenameWqni = base_Wq + "n" + std::to_string(n) + "h" + std::to_string(h);
       std::string filenameWkni = base_Wk + "n" + std::to_string(n) + "h" + std::to_string(h);
       std::string filenameWvni = base_Wv + "n" + std::to_string(n) + "h" + std::to_string(h);
       

       if(!fileExists(layerCachePaths + "/" + filenameWqni + extention) ||
        !fileExists(layerCachePaths + "/" + filenameWkni + extention) || 
        !fileExists(layerCachePaths + "/" + filenameWvni + extention)) {
          return;
        }




       LayerPerams[n].pWQ_heads[h] = loadTensor2Bin(layerCachePaths + "/" + filenameWqni + extention);
       LayerPerams[n].pWK_heads[h] = loadTensor2Bin(layerCachePaths + "/" + filenameWkni + extention);
       LayerPerams[n].pWV_heads[h] = loadTensor2Bin(layerCachePaths + "/" + filenameWvni + extention);
    } 
     
       std::string filenameWon = base_Wo + "n" + std::to_string(n);
       std::string filenameW1n = base_W1 + "n" + std::to_string(n);
       std::string filenameW2n = base_W2 + "n" + std::to_string(n);

       std::string filenameB1n = base_b1 + "n" + std::to_string(n);
       std::string filenameB2n = base_b2 + "n" + std::to_string(n);

        if(!fileExists(layerCachePaths + "/" + filenameWon + extention) ||
        !fileExists(layerCachePaths + "/" + filenameW1n + extention) || 
        !fileExists(layerCachePaths + "/" + filenameW2n + extention) || 
        !fileExists(layerCachePaths + "/" + filenameB1n + extention) ||
        !fileExists(layerCachePaths + "/" + filenameB2n + extention)) {
          return;
        }

 

       LayerPerams[n].WO = loadTensor2Bin(layerCachePaths + "/" + filenameWon + extention);
       LayerPerams[n].W1 = loadTensor2Bin(layerCachePaths + "/" + filenameW1n + extention); 
       LayerPerams[n].W2 = loadTensor2Bin(layerCachePaths + "/" + filenameW2n + extention);       
       
       LayerPerams[n].b1 = loadTensor1Bin(layerCachePaths + "/" + filenameB1n + extention);
       LayerPerams[n].b2 = loadTensor1Bin(layerCachePaths + "/" + filenameB2n + extention);

  }

  std::string base_Wp = "cacheWP_";
  std::string base_bp = "cacheBP_";
  
  std::string filenameWp = base_Wp + "0";
  std::string filenameBp = base_bp + "0";

        if(!fileExists(projectionCachePaths + "/" + filenameWp + extention) ||
        !fileExists(projectionCachePaths + "/" + filenameBp + extention)) {
          return;
        }

  WeightProjection = loadTensor2Bin(projectionCachePaths + "/" + filenameWp + extention);
  biasProjection = loadTensor1Bin(projectionCachePaths + "/" + filenameBp + extention);




}