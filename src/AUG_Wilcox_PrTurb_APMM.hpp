#ifndef AUG_Wilcox_PrTurb_APMM_hpp
#define AUG_Wilcox_PrTurb_APMM_hpp

#include "LIFE/AugContainer.hpp"
#include "LIFE/AugMaps/AdaptivePwMultiLinearMap2D.hpp"

const int nFtrs = 2;

class WilcoxPrTurbAug : public AugContainer<AdaptivePwMultiLinearMap2D<nFtrs>, nFtrs>
{
  public:

    WilcoxPrTurbAug() : AugContainer<AdaptivePwMultiLinearMap2D<nFtrs>, nFtrs>() { }

    // For convenient initialization and feature bounds checking

    // For Test 2: psi_rs_new, nu_nuT, Re_dwall, M_turb_new
    std::array<int, nFtrs> n_nodes = {41, 41};
    std::array<double, nFtrs> min_vals = {-4, -1.5};
    std::array<double, nFtrs> max_vals = {4+1e-5, 1.5+1e-5};

    inline bool checkFtr(int i) {
      return (this->ftrs[i] >= min_vals[i] && this->ftrs[i] < max_vals[i]);
    }

    inline bool checkFtr_AD(int i) {
      return (this->ftrs_AD[i].value() >= min_vals[i] && this->ftrs_AD[i].value() < max_vals[i]);
    }
    
    inline void init(
      AugVarContainer* vars_
    )
    {
      this->vars = vars_;
      // this->augMap.init("Wilcox_PrTurb", {16, 16, 16}, {0, 0, 0},
                        // {1+1e-5, 1+1e-5, 1+1e-5});
      this->augMap.init("Wilcox_PrTurb_APMM", "Adaptive_PMM_2D_v1.apmm", min_vals, max_vals);
      this->augMap.readParams();
    }

    // Wilcox 2006 model constants and one-line functions
		double sigma_k = 0.6;
		double sigma_om = 0.5;
		double beta_star = 0.09;
		double sqrt_beta_star = 0.3;
		double gamma_kom = 0.52;
		double C_lim = 0.875;
		double beta_0 = 0.0708;
		double kappa_2 = 0.16;

    double calc_gamma_kom(void) {return beta_0/beta_star - sigma_om*kappa_2/sqrt_beta_star;} 
		template<class T> T calc_sigma_d(T crossProd) {return (crossProd > 0) ? 0.125 : 0;}
		template<class T> T calc_beta_full(T f_beta) {return beta_0*f_beta;}

    // The psi-equation additions
    double psi_rs_ref = 1.0;

    // Other constants and helpers for feature calculations
		double gamma = 1.4;

    // For Test 3: psi_rs_new, nu_nuT, Re_dwall, ftr_7pt5
    double c_nu = 4.0;
    double c_T = 1.5;

    // For Test 3: psi_rs, nu_nuT, Re_dwall, M_turb
    inline void evalFtrs(int icv) {
      double kine_cv = this->vars->scal[0][0][icv];
      double omega_cv = this->vars->scal[1][0][icv];
      double psi_rs_cv = this->vars->scal[2][0][icv];

      double temp_cv = this->vars->temperature[0][icv];
      double nu_cv = this->vars->mu[0][icv] / this->vars->rho[0][icv];
      double diverg_cv = this->vars->diverg[0][icv];
      double d_cv = this->vars->wallDist[0][icv];

      // Calculate quantities used for feature 0: nu and nuTurb
      double nuT_cv = kine_cv/fmax(omega_cv, C_lim*this->vars->strMag[0][icv]/sqrt_beta_star);

      // Calculate quantities used for feature 1: wallnormal gradT
      double normal[3] = {0, 0, 0};
      // if (icv == 0) cout << "pointers test: " << this->vars->grad_enthalpy << ", " << this->vars->grad_enthalpy_AD << endl;
      // cout << "icv: " << icv << endl;
      // cout << "\t" << "wallNormal: " << this->vars->wallNormal[0][icv][0] << ", " << this->vars->wallNormal[0][icv][1] << ", " << this->vars->wallNormal[0][icv][2] << endl;
      normVec3d(normal, this->vars->wallNormal[0][icv]);
      // cout << "\t" << "normal: " << normal[0] << ", " << normal[1] << ", " << normal[2] << endl;
      // cout << "\t" << "grad_H: " << this->vars->grad_enthalpy[0][icv][0] << ", " << this->vars->grad_enthalpy[0][icv][1] << ", " << this->vars->grad_enthalpy[0][icv][2] << endl;
      double grad_h_dot_normal = this->vars->grad_enthalpy[0][icv][0]*normal[0] +
                                 this->vars->grad_enthalpy[0][icv][1]*normal[1] +
                                 this->vars->grad_enthalpy[0][icv][2]*normal[2];
      // cout << "\t" << "grad_h_dot_normal: " << grad_h_dot_normal << endl;
      double cp_air = gamma*this->vars->RoM[0][icv]/(gamma-1);
      double grad_T_dot_normal = grad_h_dot_normal/cp_air;
      // double l_turb = sqrt(kine_cv)/omega_cv/0.09;
      double quant_T = grad_T_dot_normal*d_cv/temp_cv;

      // Now calculate and populate the features
      this->ftrs[0] = c_nu*tanh(log10(1e-20 + nuT_cv/nu_cv)/c_nu);
      assert(checkFtr(0));

      this->ftrs[1] = c_T*tanh(quant_T/c_T);
      assert(checkFtr(1));
    }

    // For Test 3: psi_rs, nu_nuT, Re_dwall, M_turb
    inline void evalFtrs_AD(int icv) {
      adouble kine_cv = this->vars->scal_AD[0][0][icv];
      adouble omega_cv = this->vars->scal_AD[1][0][icv];
      adouble psi_rs_cv = this->vars->scal_AD[2][0][icv];

      adouble temp_cv = this->vars->temperature_AD[0][icv];
      adouble nu_cv = this->vars->mu_AD[0][icv] / this->vars->rho_AD[0][icv];
      adouble diverg_cv = this->vars->diverg_AD[0][icv];
      double d_cv = this->vars->wallDist[0][icv];

     // Calculate quantities used for feature 0: nu and nuTurb
      adouble nuT_cv = kine_cv/fmax(omega_cv, C_lim*this->vars->strMag_AD[0][icv]/sqrt_beta_star);

      // Calculate quantities used for feature 1: wallnormal gradT
      double normal[3] = {0, 0, 0};
      normVec3d(normal, this->vars->wallNormal[0][icv]);
      adouble grad_h_dot_normal = this->vars->grad_enthalpy_AD[0][icv][0]*normal[0] +
                                  this->vars->grad_enthalpy_AD[0][icv][1]*normal[1] +
                                  this->vars->grad_enthalpy_AD[0][icv][2]*normal[2];
      // cout << "icv: " << icv << ", grad_h_dot_normal: " << grad_h_dot_normal << endl;
      adouble cp_air = gamma*this->vars->RoM_AD[0][icv]/(gamma-1);
      adouble grad_T_dot_normal = grad_h_dot_normal/cp_air;
      // adouble l_turb = sqrt(kine_cv)/omega_cv/0.09;
      adouble quant_T = grad_T_dot_normal*d_cv/temp_cv;

      // Now calculate and populate the features
      this->ftrs_AD[0] = c_nu*tanh(log10(1e-20 + nuT_cv/nu_cv)/c_nu);
      assert(checkFtr_AD(0));

      this->ftrs_AD[1] = c_T*tanh(quant_T/c_T);
      assert(checkFtr_AD(1));
      
      // Copy values to non-AD arrays
      this->ftrs[0] = this->ftrs_AD[0].value();
			this->ftrs[1] = this->ftrs_AD[1].value();
    }
};

#endif