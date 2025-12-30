#include <iostream>

#include "model.cc"


#define dModel 512
#define Heads 8
#define dKQV (dModel/Heads)

#define MLP_hidden (dModel*4)
#define Nx 12


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
     layers_params[layer].pWQ_heads[h] = initRandomTensor2(dModel,dKQV,4745.0f + Heads*h + Nx*layer, -1.1f, 1.1f);
     layers_params[layer].pWK_heads[h] = initRandomTensor2(dModel,dKQV,8745.0f + Heads*h + Nx*layer, -1.1f, 1.1f);
     layers_params[layer].pWV_heads[h] = initRandomTensor2(dModel,dKQV,2345.0f + Heads*h + Nx*layer, -1.1f, 1.1f);
   }
   
   layers_params[layer].WO = initRandomTensor2(dModel,dModel,9987.0f + Nx*layer, -1.1f, 1.1f);
   layers_params[layer].bO = initRandomTensor2(1,dModel,7765.0f + Nx*layer, -0.1f, 1.1f)[0];

   //MLP stuff

   layers_params[layer].W1 = initRandomTensor2(dModel, MLP_hidden, 1357.0f + Nx*layer, -0.1f, 0.1f);
   layers_params[layer].b1 = initRandomTensor2(1, MLP_hidden, 2468.0f+ Nx*layer, -0.1f, 0.1f)[0];

   layers_params[layer].W2 = initRandomTensor2(MLP_hidden, dModel, 1122.0f+ Nx*layer, -0.1f, 0.1f);
   layers_params[layer].b2 = initRandomTensor2(1, dModel, 3344.0f+ Nx*layer, -0.1f, 0.1f)[0];
  }
   
  tensor2 output_projection_weights = initRandomTensor2(dModel, vocab.size(), 202421.0f, -0.1f, 0.1f);
  tensor1 output_projection_bias = initRandomTensor2(1, vocab.size(), 303421.0f, -0.1f, 0.1f)[0];

   std::string layercache_path = "tunning_cache/Layers";
   std::string projectioncache_path = "tunning_cache";
  
   loadCache(layers_params,output_projection_weights,output_projection_bias,Nx,Heads,layercache_path,projectioncache_path);
   
   Model fModel;
   fModel.vocab = vocab;
   fModel.learningRate = 1e-3;

   fModel._dModel = dModel;
   fModel._epoch = 100;
   fModel._HeadN = Heads;
   fModel._LayerN = Nx;

   fModel.layers_params = layers_params;
   fModel.output_projection_weights = output_projection_weights;
   fModel.output_projection_bias = output_projection_bias;
   
   fModel.layerCachePaths = layercache_path;
   fModel.projectionCachePaths = projectioncache_path;

   fModel.training_examples = generated_sentences;

   if(true) fModel.train(model,10,2);

   std::string prompt = "Classical Mechanics is the study of";

   fModel.generates(model,prompt,6,prompt);

      



   std::cout << prompt << "\n";
 


  return 0;
}