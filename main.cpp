#include <iostream>

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






int main() {

 //inital 01 step;   
   vocabulary vocab;
    vocab.feed_vocabulary(0);
   EmbeddingModel model(vocab.size(),dModel);

 //initial 02 step; 
/*
   tensor2 WQ = initRandomTensor2(dModel,dModel,1234.0f, -0.1f, 0.1f);
   tensor2 WK = initRandomTensor2(dModel,dModel,3424.0f, -0.1f, 0.1f);
   tensor2 WV = initRandomTensor2(dModel,dModel,4745.0f, -0.1f, 0.1f);
  
   tensor1 bQ = initRandomTensor2(1,dModel,5678.0f, -0.1f, 0.1f)[0];
   tensor1 bK = initRandomTensor2(1,dModel,51278.0f, -0.1f, 0.1f)[0];  
   tensor1 bV = initRandomTensor2(1,dModel,1078.0f, -0.1f, 0.1f)[0];
*/
  std::vector<LayerPeramaters> layers_params(Nx);

  for(int layer = 0 ; layer < Nx ; layer++) {
    //Attention stuff
  layers_params[layer].pWQ_heads = tensor3(Heads);
  layers_params[layer].pWK_heads = tensor3(Heads);
  layers_params[layer].pWV_heads = tensor3(Heads);
 

   for(int h = 0 ; h < Heads ; h++) {
     layers_params[layer].pWQ_heads[h] = initRandomTensor2(dModel,dKQV,4745.0f + Heads*h + Nx*layer, -0.1f, 0.1f);
     layers_params[layer].pWK_heads[h] = initRandomTensor2(dModel,dKQV,8745.0f + Heads*h + Nx*layer, -0.1f, 0.1f);
     layers_params[layer].pWV_heads[h] = initRandomTensor2(dModel,dKQV,2345.0f + Heads*h + Nx*layer, -0.1f, 0.1f);
   }
   
   layers_params[layer].WO = initRandomTensor2(dModel,dModel,9987.0f + Nx*layer, -0.1f, 0.1f);
   layers_params[layer].bO = initRandomTensor2(1,dModel,7765.0f + Nx*layer, -0.1f, 0.1f)[0];

   //MLP stuff

   layers_params[layer].W1 = initRandomTensor2(dModel, MLP_hidden, 1357.0f + Nx*layer, -0.1f, 0.1f);
   layers_params[layer].b1 = initRandomTensor2(1, MLP_hidden, 2468.0f+ Nx*layer, -0.1f, 0.1f)[0];

   layers_params[layer].W2 = initRandomTensor2(MLP_hidden, dModel, 1122.0f+ Nx*layer, -0.1f, 0.1f);
   layers_params[layer].b2 = initRandomTensor2(1, dModel, 3344.0f+ Nx*layer, -0.1f, 0.1f)[0];
  }
   
  tensor2 output_projection_weights = initRandomTensor2(dModel, vocab.size(), 202421.0f, -0.1f, 0.1f);
  tensor1 output_projection_bias = initRandomTensor2(1, vocab.size(), 303421.0f, -0.1f, 0.1f)[0];
  
  
  ///EXECUTION PHASE

  std::string test_sentence = "this is a sample sentence for testing the transformer";
  std::string original_sentence = test_sentence + " " + "model to generate the next words based on the given input";
     auto realtokens = tokenize(original_sentence);


  std::cout << test_sentence;

  float total_loss = 0.0f;

for(int i = 0 ; i < searchWords ; i++) {
  auto tokens = tokenize(test_sentence);

  tensor2 embedding_input(tokens.size(), tensor1(dModel,0.0f));
  
  for(int i = 0 ; i < tokens.size() ; i++) {
       size_t token_id = vocab.token_to_index(tokens[i]);
       embedding_input[i] = model.get_embedding(token_id);
  }
  
  for(int n = 0 ; n < Nx ; n++) {
      tensor2 embedding_tilda = multiHeadAttention(embedding_input,
                                              layers_params[n].pWQ_heads,
                                              layers_params[n].pWK_heads,
                                              layers_params[n].pWV_heads,
                                              layers_params[n].WO,
                                              dModel,
                                              Heads);
                                              
      //Add & Norm
      
      embedding_input = layerNorm(embedding_input + embedding_tilda, 1.0f, 0.0f);

      //Feed Forward Network
      tensor2 ffn_output = MultiLayerPreceptron(embedding_input,
                                              layers_params[n].W1,
                                              layers_params[n].b1,
                                              layers_params[n].W2,
                                              layers_params[n].b2);

      
     
      //Add & Norm
      embedding_input = layerNorm(embedding_input + ffn_output, 1.0f, 0.0f);                                       
  } 

      tensor1 targeted_embedding = embedding_input[embedding_input.size() - 1]; //last token embedding


      tensor1 logits(vocab.size(),0.0f);

      for(int j = 0 ; j < vocab.size() ; j++) {
          logits[j] = output_projection_bias[j];
          for(int k = 0 ; k < dModel ; k++) {
              logits[j] += targeted_embedding[k] * output_projection_weights[k][j];
          }
      }


      tensor1 probabilities = softMax({logits})[0];

      auto max_it = std::max_element(probabilities.begin(),probabilities.end());
      size_t index = std::distance(probabilities.begin(), max_it);
      
      
      std::string the_actual_predicted_word = vocab.index_to_token(index);
      std::string the_actual_word = realtokens[tokens.size() + i];
      

      size_t actual_index = vocab.token_to_index(the_actual_word);
      
      tensor1 actual_one_hot(vocab.size(),0.0f);
      actual_one_hot[actual_index] = 1.0f;

      for(int k = 0 ; k < actual_one_hot.size() ; k++) {
        actual_one_hot[k] = (1.0 - 0.01) * actual_one_hot[k] + 0.01 / vocab.size();
      }

      float entropy_loss = 0.0f;

      for(int k = 0 ; k < probabilities.size() ; k++) {
         entropy_loss += -actual_one_hot[k] * std::log(probabilities[k] + 1e-10f);
      }

      total_loss += entropy_loss;


      //std::cout << "Predicted next word: " << vocab.index_to_token(index) << std::endl;


      std::cout << " " << the_actual_word;

      test_sentence = test_sentence + " " + vocab.index_to_token(index);
    }

    std::cout << std::endl;
    std::cout << "Total Loss: " << total_loss <<std::endl;



  return 0;
}