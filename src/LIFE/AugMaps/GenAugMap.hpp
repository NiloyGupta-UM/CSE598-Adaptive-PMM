#ifndef GenAugMap_hpp
#define GenAugMap_hpp

#include <array>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include "mpi.h"

template <const int nFtrs>
class GenAugMap
{
  protected:

    int mpi_rank, mpi_size;
    std::string name;

  public:
    
    std::vector<double> params;
    std::vector<double> dJ_dParams;

  public:

    virtual double calculate_value(
      const std::array<double, nFtrs>& ftrs,
      double* dBeta_dFtrs
    ) = 0;

    virtual void calculate_sens(
      const std::array<double, nFtrs>& ftrs,
      const double& dJ_dBeta
    ) = 0;

  public:

    GenAugMap(
      void
    )
    {
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }

    inline void zerofySens(
      void
    )
    {
      std::fill(dJ_dParams.begin(), dJ_dParams.end(), 0.0);
    }

    inline void readParams(
      void
    )
    {
      for (int iProc=0; iProc<mpi_size; iProc++)
      {
        if (iProc==mpi_rank)
        {
          FILE *fp = fopen((name+"_params.dat").c_str(), "r");
          for(auto& param : params)
            fscanf(fp, "%le", &param);
          fclose(fp);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    inline void writeSens(
      void
    )
    {
      auto dJ_dParams_local = dJ_dParams;
      
      MPI_Allreduce(
        dJ_dParams_local.data(), dJ_dParams.data(),
        dJ_dParams.size(), MPI_DOUBLE, MPI_SUM,
        MPI_COMM_WORLD
      );

      if (mpi_rank==0)
      {
          FILE *fp = fopen((name+"_sens.dat").c_str(), "w");
          for(auto& dJ_dParam : dJ_dParams)
            fprintf(fp, "%+.10le\n", dJ_dParam);
          fclose(fp);
      }

      MPI_Barrier(MPI_COMM_WORLD);
    }
};

#endif
