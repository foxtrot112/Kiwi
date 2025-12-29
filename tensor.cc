#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


using tensor0 = float;
using tensor1 = std::vector<float>;
using tensor2 = std::vector<tensor1>;
using tensor3 = std::vector<tensor2>;


tensor1 operator+(const tensor1 &a, const tensor1 &b) { 
    tensor1 out(a.size());
    
    for(int i = 0 ; i < a.size() ; ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
};
tensor1 operator-(const tensor1 &a, const tensor1 &b) { 
    tensor1 out(a.size());
    
    for(int i = 0 ; i < a.size() ; ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
};

tensor1 operator*(const tensor1 &a, const float &b) { 
    tensor1 out(a.size());
    
    for(int i = 0 ; i < a.size() ; ++i) {
        out[i] = a[i] * b;
    }
    return out;
};

tensor1 operator*(const float &b,const tensor1 &a) { 
    tensor1 out(a.size());
    
    for(int i = 0 ; i < a.size() ; ++i) {
        out[i] = a[i] * b;
    }
    return out;
};

float dot(const tensor1 &a, const tensor1 &b) {
    float scaler_product = 0.0;

    for(int i = 0 ; i < a.size() ; ++i) {
        scaler_product += a[i]*b[i];
    }

    return scaler_product;
}


tensor1 operator*(const tensor2 &A, const tensor1 &b) {
    ///following the tensor contraction rule
    tensor1 out(A.size());
    
    for(int i = 0 ; i < A.size() ; ++i) {
        out[i] = dot(A[i],b);
    }

    return out;
}

tensor2 operator+(const tensor2 &A, const tensor2 &B) {
    tensor2 out(A.size(),tensor1(A[0].size()));

    for(int i = 0 ; i < A.size() ; i++) {
        for(int j = 0 ; j < A[0].size() ; j++) {
            out[i][j] = A[i][j] + B[i][j];
        }
    }

    return out;
}

tensor2 operator-(const tensor2 &A, const tensor2 &B) {
    tensor2 out(A.size(),tensor1(A[0].size()));

    for(int i = 0 ; i < A.size() ; i++) {
        for(int j = 0 ; j < A[0].size() ; j++) {
            out[i][j] = A[i][j] - B[i][j];
        }
    }

    return out;
}

tensor2 operator*(const tensor2 &A, const float b) {
       tensor2 out(A.size(),tensor1(A[0].size()));

    for(int i = 0 ; i < A.size() ; i++) {
        for(int j = 0 ; j < A[0].size() ; j++) {
            out[i][j] = A[i][j]*b;
        }
    }

    return out;
}

tensor2 operator*(const float b ,const tensor2 &A) {
       tensor2 out(A.size(),tensor1(A[0].size()));

    for(int i = 0 ; i < A.size() ; i++) {
        for(int j = 0 ; j < A[0].size() ; j++) {
            out[i][j] = A[i][j]*b;
        }
    }

    return out;
}

tensor2 operator*(const tensor2 &A, const tensor2 &B) {
    tensor2 out(A.size(),tensor1(B[0].size(),0.0f));
    
    for(int i = 0 ; i < A.size() ; i++) {
        for(int j = 0 ; j < B[0].size() ; j++) {
            for(int k = 0 ; k < A[0].size() ; k++) {
                out[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return out;
}

tensor2 operator*(const tensor3 &AA, const tensor1 &b) {
   tensor2 out(AA[0].size(), tensor1(AA[0][0].size()));
    
   for(int a = 0 ; a < AA.size() ; a++) {
      out = out + AA[a]*b[a];
   }
   
   return out;
}

tensor2 transpose(const tensor2 &A) {
    tensor2 out(A[0].size(), tensor1(A.size(),0.0f));
    
    for(int i = 0 ; i < A.size() ; i++) {
        for(int j = 0 ; j < A[0].size() ; j++) {
            out[j][i] = A[i][j];
        }
    }
    
    return out;
}
