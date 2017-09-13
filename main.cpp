//
//  main.cpp
//  615new1
//
//  Created by Shuting Liao on 16/12/15.
//  Copyright © 2016年 Shuting Liao. All rights reserved.
//
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <random>
#include <cstdlib>
#include<iomanip>
#include<fstream>

using namespace std;

#ifndef _Matrix_615_H_
#define _Matrix_615_H_

#include<iostream>
#include<vector>
#include<string>
#include<cstdlib>
#include<climits>
#include<cmath>
#include<fstream>
#include<iomanip>
using namespace std;

template <class T>
class Matrix615 {
public:
    vector<vector<T> > data;
    Matrix615(){
    }
    Matrix615(int nrow, int ncol, T val = 0) {
        data.resize(nrow); // make n rows
        for(int i=0; i < nrow; ++i) {
            data[i].resize(ncol,val); // make n cols with default value val
        }
    }
    
    int rowNums() {return (int)data.size();}
    int colNums() {return (data.size()==0) ? 0: (int)data[0].size();}
    void print();
    Matrix615<T> Transpose();
    Matrix615<T> operator= (const Matrix615<T>& m);
    Matrix615<T> operator+ ( Matrix615<T> & m);
    Matrix615<T> operator- ( Matrix615<T> & m);
    Matrix615<T> operator* ( Matrix615<T> & m);
    
};

template <class T>
void Matrix615<T>::print(){
    for(int i=0; i < rowNums();i++){
        for(int j=0;j < colNums(); j++){
            cout <<std::right<<setw(10)<< data[i][j] << " ";
        }
        cout << endl;
    }
}
template <class T>
Matrix615<T> Matrix615<T>::Transpose(){
    Matrix615<T> result(colNums(), rowNums(),0);
    for ( int i = 0; i < rowNums(); ++i){
        for ( int j = 0; j < colNums(); ++j){
            result.data[j][i] = data[i][j];
        }
    }
    
    return result;
    
}
template <class T>
Matrix615<T> Matrix615<T>::operator= (const Matrix615<T>& m)
{
    for ( int i = 0; i < rowNums(); ++i){
        for ( int j = 0; j < colNums(); ++j){
            data[i][j] = m.data[i][j];
        }
    }
    return *this;
}
template <class T>
Matrix615<T> Matrix615<T>::operator+ ( Matrix615<T> & m){
    Matrix615<T> a(rowNums(),m.colNums(),0);
    for(int i=0;i<rowNums();i++){
        for(int j=0;j<m.colNums();j++){
            a.data[i][j]=data[i][j]+m.data[i][j];
        }
    }
    return a;
}

template <class T>
Matrix615<T> Matrix615<T>::operator- ( Matrix615<T> & m){
    Matrix615<T> a(rowNums(),m.colNums(),0);
    for(int i=0;i<rowNums();i++){
        for(int j=0;j<m.colNums();j++){
            a.data[i][j]=data[i][j]-m.data[i][j];
        }
    }
    return a;
}

template <class T>
Matrix615<T> Matrix615<T>::operator* ( Matrix615<T> & m){
    Matrix615<T> a(rowNums(),m.colNums(),0);
    for(int i=0;i<rowNums();i++){
        for(int j=0;j<m.colNums();j++){
            for(int k=0;k<colNums();k++)
                a.data[i][j]+=data[i][k]*m.data[k][j];
        }
    }
    return a;
}
#endif

int i,j,k;


Matrix615<double> betamatrix,ya;

vector<double> stdize(Matrix615<double>& x);

Matrix615<double> X,y;
vector<double> w,t;

Matrix615<double>& oneDim(vector<double>& w, vector<double>& t, vector<int>& index, double thresh, double outerThresh,double g, double alpha, double min_frac, int nlam,int innerIter,int outerIter, int step, int reset ,int np,int lp,int n, int ncol);

vector<double> betterPathCalc(Matrix615<double>& X,Matrix615<double>& y,vector<int>& index, double alpha, double min_frac, int nlam);

void printvector(vector<int>& v);

void printvectord(vector<double>& v);

void quickSort(vector<double>& A, int p, int r);

int linNest(Matrix615<double>& X, Matrix615<double>& y, vector<int>& index, vector<int>& rangeGroupInd, int lambda1, int lambda2, vector<double>& beta,  vector<double>& eta,  vector<int>& betaIsZero, double thresh, double outerThresh,double g, double alpha, double min_frac, int nlam,int innerIter,int outerIter, int step, int reset ,int np,int lp,int n, int ncol);

Matrix615<double>& predict(vector<double>& beta, Matrix615<double>& X, int lam);

void linGradCalc(int n, vector<double>& eta, Matrix615<double>& y, double *ldot);

double linNegLogLikelihoodCalc(int n, vector<double>& eta, Matrix615<double>& y);

void linSolver(Matrix615<double>& X, Matrix615<double>& y, vector<int>& index, vector<double>& beta, vector<int>& rangeGroupInd,  int lambda1, int lambda2, double *ldot, double *nullBeta, vector<double>& eta, vector<int>& betaIsZero, int& groupChange, int* isActive, int* useGroup, double thresh, double g, int innerIter,int step, int reset ,int np,int lp,int n, int ncol);

Matrix615<double>& predict(Matrix615<double>& beta, Matrix615<double> x, Matrix615<double> y, int lam, int n, vector<double>& scale);



int main(int argc, char** argv) {
    
    mt19937 rng;
    
    vector<int> p,index;
    
    vector<double> p1;
    
    double tok;
    
    while(cin>>tok){p1.push_back(tok);}
    
    double thresh =p1[0], outerThresh=p1[1], g=p1[2], alpha=p1[3], min_frac=p1[4];
    
    int nlam=p1[5],innerIter=p1[6], outerIter=p1[7], step=p1[8], reset=p1[9], np=p1[10], lp=p1[11], n=p1[12];
    
    int ncol=np*lp;
    
    
    
    for(i=0;i<np;i++) {
        
        p.push_back(lp);
        
        for(j=0;j<lp;j++){
            
            index.push_back(i+1);
            
        }
        
    }
    
    normal_distribution <> norm(0,1);
    
    for (int i=0; i<n*ncol; i++) {
        w.push_back(norm(rng));
    }
    t.resize(n);
    for (int i=0; i<n; i++) {
        t[i]+=0.5*norm(rng);
    }
     Matrix615<double> yor(n,1,0), a(n,ncol,0), c(n,1,0);
    for (int i=0; i<n; ++i) {
        yor.data[i][0]=t[i];
        c.data[i][0]=t[i];
        
    }
    for (int i=0; i<n; i++) {
        for(int j=0; j<ncol;j++){
            
            a.data[i][j]=w[i+n*j];}
    }
    yor.print();
    a.print();
    
    vector<double> scale=stdize(yor);
     cout<<"estimated beta="<<endl;
    
    long beginTime =clock();
    
    oneDim(w, t, index, thresh, outerThresh,g, alpha,  min_frac, nlam, innerIter, outerIter, step, reset , np, lp, n, ncol);
    
    long endTime=clock();
    
    betamatrix.print();
    
    cout<<"**********"<<endl;
    
        predict(betamatrix, a, c, 1,n, scale);
    
    
    cout<<"total time:"<<1.0*(endTime-beginTime)/1000000<<"s"<<endl;
    
    return 0;
    
}









vector<double> stdize(Matrix615<double>& x){
    
    double mean=x.data[0][0], var=0;
    
    int count=2;
    
    //use west algorithm to calculate mean and variance
    
    for(int i=0;i<x.colNums();i++){
        
        for(int j=0;j<x.rowNums();j++){
            
            if(i!=0 || j!=0){
                
                var+=1.0*(count-1)/count*(x.data[j][i]-mean)*(x.data[j][i]-mean);
                
                mean=mean+(x.data[j][i]-mean)/count;
                
            }
            
            count++;
            
        }
        
        var=sqrt(var/(x.rowNums()*x.colNums()-1));
        
        for(int k=0;k<x.rowNums();k++)
            
            x.data[k][i]=(x.data[k][i]-mean)/var;
        
    }
    
    vector<double> scale;
    
    scale.push_back(mean);
    
    scale.push_back(var);
    
    return scale;
    
}



Matrix615<double>& oneDim(vector<double>& w, vector<double>& t, vector<int>& index,double thresh, double outerThresh,double g, double alpha, double min_frac, int nlam,int innerIter,int outerIter, int step, int reset ,int np,int lp,int n, int ncol){
    
    Matrix615<double> X(n,ncol,0);
    Matrix615<double> y(n,1,0);
    
    for (int i=0; i<n; i++) {
        for(int j=0; j<ncol;j++){
            
            X.data[i][j]=w[i+n*j];}
    }
    
    for (int i=0; i<n; ++i) {
        y.data[i][0]=t[i];
    }
    
    vector<double> lambdas;
    
    vector<int> rangeGroupInd;
    
    int  lambda1, lambda2;
    
    lambdas=betterPathCalc(X,y, index, alpha,  min_frac, nlam);
    
    betamatrix.data.resize((int)lambdas.size());
    
    
    vector<double> beta(np*lp,0);
    
    for(i=0;i<np;i++) {
        
        
        
        rangeGroupInd.push_back(i);}
    
    for(k=0;k<(int)lambdas.size();k++)
        
    {
        
        lambda1=lambdas[k]*alpha;
        
        lambda2=lambdas[k]*(1-alpha);
        
        vector<int> betaIsZero(np,1);
        
        vector<double> eta(n,0);
        
        linNest(X, y, index, rangeGroupInd, lambda1, lambda2,  beta,  eta, betaIsZero,  thresh, outerThresh, g,  alpha,  min_frac,  nlam, innerIter, outerIter,  step,  reset , np, lp, n, ncol);
        
        betamatrix.data[k].resize(X.rowNums());
        
        betamatrix.data[k]=beta;
       }
    
    return betamatrix;
    
}



void linGradCalc(int n, vector<double>& eta, Matrix615<double>& y, double *ldot)

{
    
    for(int i = 0; i < n; i++)
        
    {
        
        ldot[i] = (eta[i] - y.data[i][0])/n;
        
    }
    
}



double linNegLogLikelihoodCalc(int n, vector<double>& eta, Matrix615<double>& y)



{
    
    double squareSum = 0;
    
    
    
    for(int i = 0; i < n; i++)
        
    {
        
        squareSum = squareSum + pow(eta[i] - y.data[i][0], 2)/2;
        
    }
    
    
    
    return squareSum/n;
    
}



void linSolver(Matrix615<double>& X, Matrix615<double>& y, vector<int>& index, vector<double>& beta, vector<int>& rangeGroupInd,  int lambda1, int lambda2, double *ldot, double *nullBeta, vector<double>& eta, vector<int>& betaIsZero, int& groupChange, int* isActive, int* useGroup, double thresh, double g, int innerIter,int step, int reset ,int np,int lp,int n, int ncol)

{
    
    double *theta = new double[ncol];
    
    int startInd = 0;
    
    double zeroCheck = 0;
    
    double check = 0;
    
    int count = 0;
    
    double t = step;
    
    double diff = 1;
    
    double norm = 0;
    
    double uOp = 0;
    
    double Lnew = 0;
    
    double Lold = 0;
    
    double sqNormG = 0;
    
    double iProd = 0;
    
    vector<double> etaNew(n,0);
    
    vector<double> etaNull(n,0);
    
    
    for(int i = 0; i < np; i++)
        
    {
        
        if(useGroup[i] == 1)
            
        {
            
            startInd = rangeGroupInd[i]; //starting point
            
            // Setting up null gradient calc to check if group is 0
            
            for(int k = 0; k < n; k++)
                
            {
                
                etaNull[k] = eta[k];
                
                for(int j = startInd; j < rangeGroupInd[i] + lp; j++)
                    
                {
                    
                    etaNull[k] = etaNull[k] - X.data[k][j] * beta[j];//
                    
                }
                
            }
            
            
            
            // Calculating Null Gradient
            
            linGradCalc(n, etaNull, y, ldot);
            
            //Calculating the Gradient
            
            double *grad = NULL;
            
            grad = new double[lp];
            
            
            
            for(int j = 0; j < lp; j++)
                
            {
                
                grad[j] = 0;
                
                for(int k = 0; k < n; k++)
                    
                {
                    
                    grad[j] = grad[j] + X.data[k][j + rangeGroupInd[i]] * ldot[k];
                    
                }
                
                if(grad[j] < lambda1 && grad[j] > -lambda1)
                    
                {
                    
                    grad[j] = 0;
                    
                }
                
                if(grad[j] > lambda1)
                    
                {
                    
                    grad[j] = grad[j] - lambda1;
                    
                }
                
                if(grad[j] < -lambda1)
                    
                {
                    
                    grad[j] = grad[j] + lambda1;
                    
                }
                
                if(pow(grad[j],2) == pow(lambda1,2))
                    
                {
                    
                    grad[j] = 0;
                    
                }
                
                
                
            }
            
            
            
            zeroCheck = 0;
            
            for(int j = 0; j < lp; j++)
                
            {
                
                zeroCheck = zeroCheck + pow(grad[j],2);
                
            }
            
            // check if group i is zero group
            
            if(zeroCheck <= pow(lambda2,2)*lp)
                
            {
                
                if(betaIsZero[i] == 0)//beta is not zero
                    
                {
                    
                    for(int k = 0; k < n; k++)
                        
                    {
                        
                        for(int j = rangeGroupInd[i]; j < rangeGroupInd[i] + lp; j++)
                            
                        {
                            
                            eta[k] = eta[k] - X.data[k][j] * beta[j];
                            
                        }
                        
                    }
                    
                }
                
                betaIsZero[i] = 1;
                
                for(int j = 0; j < lp; j++)
                    
                {
                    
                    beta[j + rangeGroupInd[i]] = 0;
                    
                }
                
            }
            
            // if group i is nonzero group
            
            else
                
            {
                
                if(isActive[i] == 0)
                    
                {
                    
                    groupChange = 1;
                    
                }
                
                isActive[i] = 1;
                
                
                
                for(int k = 0; k < ncol; k++)
                    
                {
                    
                    theta[k] = beta[k];
                    
                }
                
                
                
                betaIsZero[i] = 0;
                
                double *z = NULL;
                
                z = new double[lp];
                
                double *U = NULL;
                
                U = new double[lp];
                
                double *G = NULL;
                
                G = new double[lp];
                
                double *betaNew = NULL;
                
                betaNew = new double[ncol];
                
                
                
                count = 0;
                
                check = 100000;
                
                
                // the limit of convergence for inner loop
                
                while(count <= innerIter && check > thresh)
                    
                {
                    
                    
                    
                    count++;
                    
                    
                    
                    linGradCalc(n, eta, y ,ldot);
                    
                    //Calculatine gradient
                    
                    for(int j = 0; j < lp; j++)
                        
                    {
                        
                        grad[j] = 0;
                        
                        for(int k = 0; k < n; k++)
                            
                        {
                            
                            grad[j] = grad[j] + X.data[k][j + rangeGroupInd[i]] * ldot[k];
                            
                        }
                        
                        
                        
                    }
                    
                    
                    
                    diff = -1;
                    
                    t =1;
                    Lold = linNegLogLikelihoodCalc(n, eta, y);
                    
                    
                    
                    // Back-tracking
                    while(diff < 0)
                    {
                        
                        for(int j = 0; j < lp; j++)
                            
                        {
                            
                            // compute gradient
                            
                            z[j] = beta[j + rangeGroupInd[i]] - t * grad[j];
                            
                            if(z[j] < lambda1 * t && z[j] > -lambda1 * t)
                                
                            {
                                
                                z[j] = 0;
                                
                            }
                            
                            if(z[j] > lambda1 * t)
                                
                            {
                                
                                z[j] = z[j] - lambda1 * t;
                                
                            }
                            
                            if(z[j] < -lambda1 * t)
                                
                            {
                                
                                z[j] = z[j] + lambda1 * t;
                                
                            }
                            
                        }
                        
                        //compute for the inequation for inner loop
                        
                        norm = 0;
                        
                        for(int j = 0; j < lp; j++)
                            
                        {
                            
                            norm = norm + pow(z[j],2);
                            
                        }
                        
                        norm = sqrt(norm);
                        
                        uOp = (1 - lambda2*sqrt(lp)*t/norm);
                        
                        if(uOp < 0)
                            
                        {
                            
                            uOp = 0;
                            
                        }
                        
                        
                        
                        for(int j = 0; j < lp; j++)
                            
                        {
                            
                            U[j] = uOp*z[j];
                            
                            G[j] = 1/t *(beta[j + rangeGroupInd[i]] - U[j]);
                            
                            
                            
                        }
                        
                        
                        
                        // Setting up betaNew and etaNew in direction of Grad for descent step
                        
                        
                        
                        for(int k = 0; k < n; k++)
                            
                        {
                            
                            etaNew[k] = eta[k];
                            
                            for(int j = 0; j < lp; j++)
                                
                            {
                                
                                etaNew[k] = etaNew[k] - t*G[j] * X.data[k][rangeGroupInd[i] + j];
                                
                            }
                            
                        }
                        
                        Lnew = linNegLogLikelihoodCalc(n, etaNew, y);
                        
                        sqNormG = 0;
                        
                        iProd = 0;
                        
                        
                        
                        for(int j = 0; j < lp; j++)
                            
                        {
                            
                            sqNormG = sqNormG + pow(G[j],2);
                            
                            iProd = iProd + grad[j] * G[j];
                            
                        }
                        
                        
                        
                        diff = Lold - Lnew - t * iProd + t/2 * sqNormG;
                        
                        
                        
                        t = t * g;
                        
                    }
                    
                    t = t / g;
                    
                    
                    
                    check = 0;
                    
                    
                    
                    for(int j = 0; j < lp; j++)
                        
                    {
                        
                        check = check + fabs(theta[j + rangeGroupInd[i]] - U[j]);
                        
                        for(int k = 0; k < n; k++)
                            
                        {
                            
                            eta[k] = eta[k] - X.data[k] [j + rangeGroupInd[i]]*beta[j + rangeGroupInd[i]];
                            
                        }
                        
                        // aaply Nesterov-style momentum updates—this allows us to leverage some higher order information while only calculating gradients.
                        
                        beta[j + rangeGroupInd[i]] = U[j] + count%reset/(count%reset+3) * (U[j] - theta[j + rangeGroupInd[i]]);
                        
                        // update theta by U
                        
                        theta[j + rangeGroupInd[i]] = U[j];
                        
                        // to get new eta for next loop
                        
                        for(int k = 0; k < n; k++)
                            
                        {
                            
                            eta[k] = eta[k] + X.data[k][j + rangeGroupInd[i]]*beta[j + rangeGroupInd[i]];
                            
                        }
                        
                    }
                    
                }
                
                delete [] z;
                
                delete [] U;
                
                delete [] G;
                
                delete [] betaNew;
                
            }
            
            delete [] grad;
            
        }
        
    }
    
    
    
    delete [] theta;
    
}



int linNest(Matrix615<double>& X, Matrix615<double>& y, vector<int>& index, vector<int>& rangeGroupInd, int lambda1, int lambda2, vector<double>& beta,  vector<double>& eta,  vector<int>& betaIsZero, double thresh, double outerThresh,double g, double alpha, double min_frac, int nlam,int innerIter,int outerIter, int step, int reset ,int np,int lp,int n, int ncol)

{
    
    double* prob = NULL;
    
    prob = new double[n];
    
    double* nullBeta = NULL;
    
    nullBeta = new double[ncol];
    
    
    
    int p = ncol;
    
    double *ldot = NULL;
    
    ldot = new double[n];
    
    int groupChange = 1;
    
    int* isActive = NULL;
    
    isActive = new int[np];
    
    int* useGroup = NULL;
    
    useGroup = new int[np];
    
    int* tempIsActive = NULL;
    
    tempIsActive = new int[np];
    
    
    
    for(int i = 0; i < np; i++)
        
    {
        
        isActive[i] = 0;
        
        useGroup[i] = 1;
        
    }
    
    
    
    
    
    int outermostCounter = 0;
    
    double outermostCheck = 100000;
    
    double* outerOldBeta = NULL;
    
    outerOldBeta = new double[p];
    
    
    
    while(groupChange == 1)
        
    {
        
        groupChange = 0;
        
        
        
        
        
        linSolver(X, y, index, beta, rangeGroupInd, lambda1, lambda2,  ldot, nullBeta, eta, betaIsZero, groupChange, isActive, useGroup, thresh, g, innerIter, step, reset , np, lp, n,  ncol);
        
        // the limit of convergence for outer loop
        
        while(outermostCounter < outerIter && outermostCheck > outerThresh)
            
        {
            
            outermostCounter ++;
            
            for(int i = 0; i < p; i++)
                
            {
                
                outerOldBeta[i] = beta[i];
                
            }
            
            
            
            for(int i = 0; i < np; i++)
                
            {
                
                tempIsActive[i] = isActive[i];
                
            }
            
            linSolver(X, y, index, beta, rangeGroupInd, lambda1, lambda2,  ldot, nullBeta, eta, betaIsZero, groupChange, isActive, useGroup, thresh, g, innerIter, step, reset , np, lp, n,  ncol );
            
            outermostCheck = 0;
            
            for(int i = 0; i < p; i++)
                
            {
                
                outermostCheck = outermostCheck + fabs(outerOldBeta[i] - beta[i]);
                
            }
            
            
            
        }}
    
    
    
    delete [] nullBeta;
    
    delete [] outerOldBeta;
    
    delete [] ldot;
    
    delete [] isActive;
    
    delete [] useGroup;
    
    delete [] tempIsActive;
    
    delete [] prob;
    
    
    
    return 1;
    
}







//////////////////////////////////





vector<double> betterPathCalc(Matrix615<double>& X,Matrix615<double>& y,vector<int>& index, double alpha, double min_frac, int nlam){
    
    vector<double> lambdas;
    
    int n=X.rowNums(),num_groups=index[index.size()-1],group_length=X.colNums()/num_groups,i;
    
    double lambda_max[num_groups],our_range[2];
    
    vector<double> our_cors,norms;
    
    for(i=0;i<num_groups;i++){
        
        Matrix615<double> X_fit(n,group_length,0);
        
        for(j=0;j<n;j++){
            for(k=0;k<group_length;k++)
                
                X_fit.data[j][k]=X.data[j][k+i*group_length];
            
        }
        
        
        Matrix615<double> cors(X_fit.colNums(),1,0),XT_fit(X_fit.colNums(),X_fit.rowNums(),0);
        
        XT_fit=X_fit.Transpose();
        
        cors=XT_fit*y;
        
        vector<double> ord_cors,lam;
        
        for(j=0;j<cors.rowNums();j++){
            
            if(cors.data[j][0]<0) cors.data[j][0]*=-1;
            
            ord_cors.push_back(cors.data[j][0]);
            
            lam.push_back(cors.data[j][0]/alpha);
            
        }
        
        quickSort(ord_cors,0,cors.rowNums());
        
        
        if(cors.rowNums()>1){
            
            for(j=1;j<group_length;j++){
                
                int sum=0;
                
                for(k=0;k<j;k++){
                    
                    sum+=(ord_cors[k]-ord_cors[j])*(ord_cors[k]-ord_cors[j]);
                    
                }
                
                norms.push_back(sqrt(sum));
                
            }
            
            if(norms[0]>lam[1]*(1-alpha)*sqrt(group_length)){
                
                our_cors.push_back(ord_cors[0]);
                
                our_range[0]=lam[0];
                
                our_range[1]=lam[1];
                
            }
            
            else{
                
                if(norms[group_length]<=lam[group_length+1] * (1-alpha)*sqrt(group_length)){
                    
                    for(int j=0;j<group_length;j++) our_cors.push_back(ord_cors[j]);
                    
                    our_range[0]=0;
                    
                    our_range[1]=lam[group_length-1];
                    
                }
                
                else{
                    
                    int j;
                    
                    double lam1=lam[lam.size()-1]* (1-alpha) * sqrt(group_length);
                    
                    for(j=(int)lam.size()-1;j>=0;j--){
                        
                        if(norms[j]<=lam1) break;
                        
                    }
                    
                    for(int k=0;k<=j;k++){
                        
                        our_cors.push_back(ord_cors[k]);
                        
                        our_range[0]=lam[j];
                        
                        our_range[1]=lam[j+1];
                        
                    }
                    
                }
                
            }
            
            double A=our_cors.size()*alpha*alpha-(1-alpha)*(1-alpha)*group_length,B=0,C=0;
            
            for(int j=0;j<our_cors.size();j++){
                
                B+=-2*alpha*our_cors[j];
                
                C+=our_cors[j]*our_cors[j];
                
            }
            
            double lams0=sqrt(B*B-4*A*C);//lam1=(-B-sqrt(B*B-4*A*C))/(2*A);
            
            
            
            if(lams0<our_range[0])lambda_max[i]=our_range[0];
            
            else lambda_max[i]=lams0;
            
            ord_cors.clear();norms.clear();lam.clear();
            
        }
        
        else if(cors.rowNums()==1)lambda_max[i]=ord_cors[0];
        
    }
    
    double max_lam=lambda_max[0],min_lam=min_frac*max_lam,temp=exp((log(max_lam)-log(min_lam))/(nlam-1));
    
    //temp=pow(max_lam,1/(nlam-1))/min_lam;
    
    lambdas.push_back(max_lam);
    
    for(i=1;i<nlam;i++){
        
        lambdas.push_back(lambdas[i-1]/temp);
        
    }
    
    return lambdas;
    
}



void printvector(vector<int>& v){
    
    for(i=0;i<v.size();i++)
        
        cout<<v[i]<<" ";
    
    cout<<endl;
    
}



void printvectord(vector<double>& v){
    
    for(i=0;i<v.size();i++)
        
        cout<<v[i]<<" ";
    
    cout<<endl;
    
}



void quickSort(vector<double>& A, int p, int r) {
    
    if ( p < r ) { // immediately terminate if subarray size is 1
        
        double piv = A[r]; // take a pivot value
        
        int i = p-1; // p-i-1 is the # elements < piv among A[p..j]
        
        double tmp;
        
        for(int j=p; j < r; ++j) {
            
            if ( A[j] > piv ) { // if smaller value is found, increase q (=i+1)
                
                ++i;
                
                tmp = A[i]; A[i] = A[j]; A[j] = tmp; // swap A[i] and A[j]
                
            }
            
        }
        
        A[r] = A[i+1]; A[i+1] = piv; // swap A[i+1] and A[r]
        
        quickSort(A, p, i);
        
        quickSort(A, i+2, r);
        
    }
    
}



Matrix615<double>& predict(Matrix615<double>& beta, Matrix615<double> x, Matrix615<double> y, int lam, int n, vector<double>& scale){
    
    cout<<"Predicting value y..."<<endl;
    
    Matrix615<double> b(beta.colNums(),1,0),xt(x.colNums(),x.rowNums(),0);
    
    double mean=scale[0],var=scale[1];
    
    ya.data.resize(n);
    
    for(i=0;i<n;i++){
        
        ya.data[i].resize(1);
        
    }
   
    for(i=0;i<b.rowNums();i++){
        
        b.data[i][0]=beta.data[lam-1][i];
        
    }
        
    for(i=0;i<x.rowNums();i++){
        
        for(j=0;j<x.colNums();j++)
            
            x.data[i][j]=(x.data[i][j]-mean)/var;
        
    }
    
    ya=x*b;
    
    for(i=0;i<ya.rowNums();i++)
        
        ya.data[i][0]+=mean;
    
    double TSS=0,RSS=0,rmse=0;
    
    for(i=0;i<x.rowNums();i++){
        
        RSS+=pow(ya.data[i][0]-y.data[i][0],2);
        
        TSS+=pow(y.data[i][0]-scale[0],2);
        
        rmse+=pow(y.data[i][0]-ya.data[i][0],2);
        
    }
      cout<<"fitted y:"<<endl;
    
    
    ya.print();
    
    cout<<"R^2  = "<<1-RSS/TSS<<endl;
    
    cout<<"rmse = "<<rmse<<endl;
    
    return ya;
    
}
//}