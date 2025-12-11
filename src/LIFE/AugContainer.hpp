#ifndef AugContainer_hpp
#define AugContainer_hpp

#include "AugMaps/GenAugMap.hpp"
#include "macros.h"

typedef  double  vec3_t[3];
typedef adouble avec3_t[3];

struct AugVarContainer
{
  int *ncv;

  double **rho;
  vec3_t **rhou;
  double **rhoE;
  double **diverg;
  double **strMag;
  double **vortMag;
  double **mu;
  double **temperature;
  double **wallDist;
  vec3_t **wallNormal;
  vec3_t **grad_enthalpy;
  double **marker;
  double **RoM;
  std::vector<double**> scal;
  std::vector<vec3_t**> grad_scal;

  adouble **rho_AD;
  avec3_t **rhou_AD;
  adouble **rhoE_AD;
  adouble **diverg_AD;
  adouble **strMag_AD;
  adouble **vortMag_AD;
  adouble **mu_AD;
  adouble **temperature_AD;
  avec3_t **grad_enthalpy_AD;
  adouble **marker_AD;
  adouble **RoM_AD;
  std::vector<adouble**> scal_AD;
  std::vector<avec3_t**> grad_scal_AD;
};

template <class AugMap, const int nFtrs>
struct AugContainer
{
  AugMap augMap;
  double *psi_kine, *psi_omega, *psi_psi_rs;
  unsigned int *dR_dBeta_rind, *dR_dBeta_cind; 
  double *dR_dBeta_values;
  double *dJ_dBeta;
  AugVarContainer *vars;

  double beta;
  adouble beta_AD;
  std::array<double, nFtrs> ftrs;
  std::array<adouble, nFtrs> ftrs_AD;
  std::array<double, nFtrs> dBeta_dFtrs;

  AugContainer(
    void
  )
  {
    psi_kine      = NULL;
    psi_omega      = NULL;
    psi_psi_rs      = NULL;
    dR_dBeta_rind  = NULL;
    dR_dBeta_cind  = NULL;
    dR_dBeta_values  = NULL;
    dJ_dBeta = NULL;
    vars     = NULL;
  }

  virtual void init(
    AugVarContainer* vars_
  ) = 0;

  virtual void evalFtrs(
    int icv
  ) = 0;

  virtual void evalFtrs_AD(
    int icv
  ) = 0;

  inline void calculate(
    int icv
  )
  { 
    evalFtrs(icv);
    beta = augMap.calculate_value(ftrs);
  }

  inline void calculate_AD(
    int icv
  )
  {
    evalFtrs_AD(icv);
    beta = augMap.calculate_value(ftrs, dBeta_dFtrs.data());
    beta_AD = beta;
    for(int iFtr=0; iFtr<nFtrs; iFtr++)
    {
      beta_AD = beta_AD +
        (ftrs_AD[iFtr] - ftrs[iFtr]) * dBeta_dFtrs[iFtr];
    }
  }

  inline void writeSens(
    void
  )
  {
    augMap.zerofySens();
    
    for(int icv=0; icv<vars->ncv[0]; icv++)
    {
      evalFtrs(icv);
      // augMap.calculate_sens(ftrs, psi[icv]*dR_dBeta[icv]);
      augMap.calculate_sens(ftrs, dJ_dBeta[icv]);
    }

    augMap.writeSens();
  }
};

#endif
