#ifndef PwMultiLinAugMap_hpp
#define PwMultiLinAugMap_hpp

#include <cstdio>
#include "../GenAugMap.hpp"
#include "MultiLinearFunction.hpp"

template <const int nFtrs>
class PwMultiLinAugMap : public GenAugMap<nFtrs>
{
  protected:

    std::array<int, nFtrs> nNodes;
    std::array<int, nFtrs> offsets;
    std::array<int, 1<<nFtrs> C_ind;
    std::array<double, nFtrs> minFtrs;
    std::array<double, nFtrs> maxFtrs;
    MultiLinearFunction<nFtrs> func;

  public:

    PwMultiLinAugMap() : GenAugMap<nFtrs>() { }

    inline void init(
      const std::string& name_,
      const std::array<int, nFtrs>& nNodes_,
      const std::array<double, nFtrs>& minFtrs_,
      const std::array<double, nFtrs>& maxFtrs_
    )
    {
      this->name = name_;
      nNodes = nNodes_;
      minFtrs = minFtrs_;
      maxFtrs = maxFtrs_;
      
      int nParams = 1;
      for(auto& nNodesFtr : nNodes)
        nParams *= nNodesFtr;
      this->params.resize(nParams);
      this->dJ_dParams.resize(nParams);

      offsets[nFtrs-1] = 1;
      for(int iFtr=nFtrs-2; iFtr>=0; iFtr--)
        offsets[iFtr] = nNodes[iFtr+1] * offsets[iFtr+1];

      for(int iFtr=0; iFtr<nFtrs; iFtr++)
        func.h[iFtr] = (maxFtrs[iFtr] - minFtrs[iFtr]) / (nNodes[iFtr]-1);
    }

    inline double calculate_value(
      const std::array<double, nFtrs>& ftrs,
      double* dBeta_dFtrs = nullptr
    )
    {
      std::array<double, nFtrs> ftrs0; C_ind[0] = 0;

      for(int iFtr=0; iFtr<nFtrs; iFtr++)
      {
        int ind = (int)((ftrs[iFtr]-minFtrs[iFtr])/func.h[iFtr]);
        ftrs0[iFtr] = ind * func.h[iFtr] + minFtrs[iFtr];
        C_ind[0] += ind * offsets[iFtr];
      }
      
      for(int iC=0; iC<C_ind.size(); iC++)
      {
        C_ind[iC] = C_ind[0];

        for(int iFtr=0; iFtr<nFtrs; iFtr++) if((iC>>iFtr)&1)
        {
          C_ind[iC] += offsets[iFtr];
        }

        func.C[iC] = this->params[C_ind[iC]];
      }

      return func.calculate(ftrs, ftrs0, dBeta_dFtrs);
    }

    inline void calculate_sens(
      const std::array<double, nFtrs>& ftrs,
      const double& dJ_dBeta
    )
    {
      double beta = calculate_value(ftrs);
      
      for(int iC=0; iC<C_ind.size(); iC++)
      {
        this->dJ_dParams[C_ind[iC]] += dJ_dBeta * func.dC[iC];
      }
    }
};

#endif
