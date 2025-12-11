#ifndef MultiLinearFunction_hpp
#define MultiLinearFunction_hpp

#include <array>
#include <algorithm>

template <const int nDim>
class MultiLinearFunction
{
  public:
    
    std::array<double, nDim> h;
    std::array<double, 1<<nDim> C, dC;

  public:

    inline double calculate(
      const std::array<double, nDim>& x,
      const std::array<double, nDim>& x0,
      double* dx = nullptr
    )
    {
      double beta = 0.0;
      std::array<double, nDim> local_basis;
      if(dx!=nullptr) std::fill(dx, dx+nDim, 0.0);

      for(int iC=0; iC<(1<<nDim); iC++)
      {
        double basis = 1.0;
        
        for(int iDim=0; iDim<nDim; iDim++)
        {
          local_basis[iDim] = (x[iDim] - x0[iDim]) / h[iDim];
          if(!((iC>>iDim)&1))
            local_basis[iDim] = 1.0 - local_basis[iDim];
          basis *= local_basis[iDim];
        }
        
        if(dx!=nullptr)
        {
          for(int iDim=0; iDim<nDim; iDim++)
          {
            double d_basis = 1.0;

            for(int jDim=0; jDim<nDim; jDim++)
            {
              if(iDim==jDim)
                d_basis *= (((iC>>iDim)&1) ? 1.0 : -1.0) / h[jDim];
              else
                d_basis *= local_basis[jDim];
            }

            dx[iDim] += C[iC] * d_basis;
          }
        }

        beta += C[iC] * basis;
        dC[iC] = basis;
      }

      return beta;
    }
};

#endif
