#ifndef RANSTURB_H
#define RANSTURB_H

#include "UgpWithCvCompFlow.h"
#include "AUG_Wilcox_PrTurb_APMM.hpp"
// #include "AUG_Wilcox_PrTurb.hpp"

// Note incorrect calculation for production limiter (should just be to kine src, not both k and omega)

class RansTurb : virtual public UgpWithCvCompFlow
{
	public:

		double *muT, *muLam, *wallDist, (*wallNormal)[3], *marker;
		double (*grad_kine)[3];
		double *omega, (*grad_omega)[3];
		double sigma_k, sigma_om, beta_star, sqrt_beta_star, gamma_kom, C_lim, beta_0, kappa_2;
		double *psi_rs, (*grad_psi_rs)[3];
		double psi_rs_ref, ls_a;
		// double *beta;
		std::vector<MarkerData> inletData;
		std::vector<std::string> turbMods;
		bool use_mod_prod = false, use_klim = false, use_Durbin = false;

		AugVarContainer augVars;
    	WilcoxPrTurbAug aug;
	
	public: 

		virtual ~RansTurb(void) {}

		RansTurb(void) {
			if(mpi_rank==0) std::cout<<"RansTurb()"<<std::endl;
			turbModel = KOM;

			if(checkParam("TURB_MODS")) {
				Param *param = getParam("TURB_MODS");
				for(int i=1; i<param->getSize(); i++)
				turbMods.push_back(param->getString(i));

				if (std::any_of(turbMods.begin(), turbMods.end(), [](std::string s){return s=="MOD_PROD";})) {PRINT("Using modified production \n"); use_mod_prod = true;}
				if (std::any_of(turbMods.begin(), turbMods.end(), [](std::string s){return s=="KLIM";})) {PRINT("Using production limiter \n"); use_klim = true;}
				if (std::any_of(turbMods.begin(), turbMods.end(), [](std::string s){return s=="DURBIN";})) {PRINT("Using Durbin correction \n"); use_Durbin = true;}
			}

			registerDefaultVariables();
			registerModelEquations();

			registerBetaAugVec(1);

			MPI_Barrier(mpi_comm);
			DEBUG_PRINT(mpi_rank, "Starting linking for LIFE stuff\n")

			// Linking for all the LIFE stuff
			augVars.ncv = &ncv;

			augVars.rho = &rho;
			DEBUG_PRINT(mpi_rank, "rho pointer linked\n")
			augVars.rhou = &rhou;
			DEBUG_PRINT(mpi_rank, "rhou pointer linked\n")
			augVars.rhoE = &rhoE;
			DEBUG_PRINT(mpi_rank, "rhoE pointer linked\n")
			augVars.diverg = &diverg;
			augVars.strMag = &strMag;
			augVars.vortMag = &vortMag;
			augVars.mu = &muLam;
			augVars.temperature = &temp;
			augVars.wallDist = &wallDist;
			augVars.wallNormal = &wallNormal;
			augVars.marker = &marker;
			augVars.grad_enthalpy = &grad_enthalpy;
			// cout << "pointers test: " << augVars.grad_enthalpy << ", " << &grad_enthalpy << endl;
			augVars.RoM = &RoM;
			augVars.scal.resize(3);
			augVars.grad_scal.resize(3);
			augVars.scal[0] = &kine;
			augVars.grad_scal[0] = &grad_kine;
			augVars.scal[1] = &omega;
			augVars.grad_scal[1] = &grad_omega;
			augVars.scal[2] = &psi_rs;
			augVars.grad_scal[2] = &grad_psi_rs;


			#ifndef ADJOINT_MODE
			MPI_Barrier(mpi_comm);
				aug.init(&augVars);
			#endif
		}

	void registerDefaultVariables(void) {
		strMag		= NULL; registerScalar(strMag,		"strMag",		CV_DATA);
		vortMag		= NULL; registerScalar(vortMag,		"vortMag",		CV_DATA);
		diverg		= NULL; registerScalar(diverg,		"diverg",		CV_DATA);
		muT			= NULL; registerScalar(muT,			"muT",			CV_DATA);
		muLam		= NULL; registerScalar(muLam,		"muLam",		CV_DATA);
		wallDist	= NULL; registerScalar(wallDist,	"wallDist",		CV_DATA);
		wallNormal	= NULL; registerVector(wallNormal,	"wallNormal",	CV_DATA);
		marker		= NULL; registerScalar(marker,		"marker",		CV_DATA);
		// beta        = NULL; registerScalar(beta,		"beta",			CV_DATA);
    }

	void registerModelEquations(void) {
    	registerModelEquation("kine", 1e-10, 1e10);
		registerModelEquation("omega", 1e-10, 1e13);
		registerModelEquation("psi_rs", 1e-10, 1e2);
    }

	void registerModelEquation(const std::string &name, const double &lb, const double &ub) {
		ScalarTranspEq *eq    = registerScalarTransport(name.c_str(), CV_DATA);
		eq->relax             = getDoubleParam(("RELAX_"+name).c_str(),"0.4");
		eq->phiZero           = 1e-9;
		eq->phiZeroRel        = 1e-4;
		eq->phiMaxiter        = 200;
		eq->lowerBound        = lb;
		eq->upperBound        = ub;
	}

	vector<double> betaAug_baseline={0.0};

	void initialHookScalarRansTurbModel(void) {
		if(mpi_rank==0) printf("initialHookScalarRansTurbModel()\n");
		
		initializeWallDistance();
		updateCvDataG1G2(wallDist, REPLACE_DATA);
		
		if(mpi_rank==0) printf("Initializing scalar variables and gradients\n");
		initializeModelEquations();
		
		initializeCoefficients();
		initializeBetaAugVec(betaAug_baseline);

		if(mpi_rank==0) printf("initialHookScalarRansTurbModel() done\n");
    }

	void initializeCoefficients(void) {
		sigma_k		= getDoubleParam("sigma_k", "0.6");
		sigma_om	= getDoubleParam("sigma_om", "0.5");
		beta_star	= getDoubleParam("beta_star", "0.09");
		sqrt_beta_star = sqrt(beta_star);
		gamma_kom	= getDoubleParam("gamma_kom", "0.52");
		C_lim		= getDoubleParam("C_lim", "0.875");
		beta_0		= getDoubleParam("beta_0", "0.0708");
		kappa_2		= getDoubleParam("kappa_2", "0.16");

		psi_rs_ref 	= getDoubleParam("psi_rs_ref", "1.0");
		ls_a 		= getDoubleParam("ls_a", "1.0");
	}

	void initializeWallDistance(void) {
		if(checkParam("NO_WALLDIST")) {
			for(int icv=0; icv<ncv_gg; icv++) wallDist[icv] = 1e5;
		}
		else {
			for(int icv=0; icv<ncv; icv++) wallDist[icv] = 0.0;
			calcWallDistanceNormals(NULL, wallDist, wallNormal);
		}
    }

	void initializeModelEquations(void) {
      	initializeModelEquation("kine", &kine, &grad_kine);
		initializeModelEquation("omega", &omega, &grad_omega);
		initializeModelEquation("psi_rs", &psi_rs, &grad_psi_rs);
    }

	void initializeModelEquation(std::string name, double **var, double (**grad_var)[3]) {
		Param *pmy;
		bool chk = getParam(pmy, name+"_INITIAL");
		if(!chk && mpi_rank==0){ printf(("Could not find "+name+"_INITIAL").c_str()); throw(-1); }
		if(mpi_rank==0) printf(("Initializing scalar "+name).c_str());
		double value = pmy->getDouble(1);
		ScalarTranspEq *eq = getScalarTransportData(name.c_str());
		(*var) = eq->phi;
		(*grad_var) = eq->grad_phi;
		if(!checkScalarFlag((char*)(name.c_str())))
			for(int icv=0; icv<ncv; icv++) (*var)[icv] = value;
	}

	void getInterpolationFactors(int ifa, int &icv0, int &icv1, double &w0, double &w1) {
		icv0 = cvofa[ifa][0];
		icv1 = cvofa[ifa][1];
		double dx0[3], dx1[3];
		vecMinVec3d(dx0, x_fa[ifa], x_cv[icv0]);
		vecMinVec3d(dx1, x_fa[ifa], x_cv[icv1]);
		w0 = sqrt(vecDotVec3d(dx0, dx0));
		w1 = sqrt(vecDotVec3d(dx1, dx1));
		double ws = w0 + w1; w0 /= ws; w1 /= ws;
	}

	// Standard Wilcox2006 equations
	double calc_gamma_kom(void) {return beta_0/beta_star - sigma_om*kappa_2/sqrt_beta_star;} 
	double calc_sigma_d(double crossProd) {return (crossProd > 0) ? 0.125 : 0;}
	double calc_beta_full(double f_beta) {return beta_0*f_beta;}

	void calcMarkerFunction(void) {
		double s_vec[3], g_vec[3], vel_mag, g_mag, f_mark;

		for (int icv=0; icv<ncv; icv++) {
			// compute (u_k u_k)^0.5
			vel_mag = sqrt(vel[icv][0]*vel[icv][0] + vel[icv][1]*vel[icv][1] + vel[icv][2]*vel[icv][2]);

			// compute u_i / (u_k u_k)^0.5
			s_vec[0] = vel[icv][0] / vel_mag;
			s_vec[1] = vel[icv][1] / vel_mag;
			s_vec[2] = vel[icv][2] / vel_mag;

			// compute g_j vector
			for (int j=0; j < 3; j++) {
				g_vec[j] = 0.0;
				for (int i=0; i < 3; i++) g_vec[j] += s_vec[i]*grad_u[icv][i][j];
			}

			g_mag = max(sqrt(g_vec[0]*g_vec[0] + g_vec[1]*g_vec[1] + g_vec[2]*g_vec[2]),1.e-12);

			// compute f = |g_j s_j| / (g_k g_k)^0.5
			f_mark = 0.0;
			// for (int i=0; i < 3; i++) f_mark += fabs(g_vec[i]*s_vec[i]);
			for (int i=0; i<3; i++) f_mark += g_vec[i]*s_vec[i];				// <-- This should be correct dot product
			f_mark = fabs(f_mark);
			f_mark /= g_mag;

			// marker definition
			marker[icv] = f_mark * kine[icv] / (vel_mag * vel_mag);
		}
	}

	void calcRansTurbViscMuet(void) {
		calcGradVel();
		calcStrainRateAndDivergence();
		calcVorticity();
		calcMarkerFunction();

		double rho_fa, kine_fa, omega_fa, omega_hat, strMag_fa;
		int icv0, icv1; double w0, w1;

		for(int ifa=nfa_b; ifa<nfa; ifa++) {
			getInterpolationFactors(ifa, icv0, icv1, w0, w1);

			rho_fa = w1*rho[icv0] + w0*rho[icv1];
			kine_fa = w1*kine[icv0] + w0*kine[icv1];
			omega_fa = w1*omega[icv0] + w0*omega[icv1];
			strMag_fa = w1*strMag[icv0] + w0*strMag[icv1];
			omega_hat = fmax(omega_fa, C_lim*strMag_fa/sqrt_beta_star);
			
			mut_fa[ifa] = rho_fa*kine_fa/omega_hat;
		}

		for(list<FaZone>::iterator ZZ=faZoneList.begin(); ZZ!=faZoneList.end(); ZZ++) {
			if(ZZ->getKind() == FA_ZONE_BOUNDARY) {
				if(zoneIsWall(ZZ->getName())) {
        			for(int ifa=ZZ->ifa_f; ifa<=ZZ->ifa_l; ifa++) {
						mut_fa[ifa] = 0.0;  
					}
				}

				if(!zoneIsWall(ZZ->getName())) {
					for(int ifa=ZZ->ifa_f; ifa<=ZZ->ifa_l; ifa++) {
						icv0 = cvofa[ifa][0];
						icv1 = cvofa[ifa][1];
						omega_hat = fmax(omega[icv1], C_lim*strMag[icv0]/sqrt_beta_star);
						mut_fa[ifa] = rho[icv1]*kine[icv1]/omega_hat;
					}
				}
			}
		}

		for (int icv=0; icv<ncv; icv++) {
			// muT[icv] = InterpolateAtCellCenterFromFaceValues(mut_fa, icv);
			muT[icv] = rho[icv]*kine[icv]/fmax(omega[icv], C_lim*strMag[icv]/sqrt_beta_star);
		}
	}

	void calcTurbulentPrandtlNumber() {
		// Need to do some prep for all the LIFE stuff
		updateCvDataG1G2(enthalpy, REPLACE_DATA);
		calcCv2Grad(grad_enthalpy, enthalpy, limiterNavierS, enthalpy, epsilonSDWLS);       // enthalpy gradients
		updateCvDataG1(grad_enthalpy, REPLACE_ROTATE_DATA);		// <-- Needed as never done in backend and will get NaNs otherwise
		updateCvDataG1G2(RoM, REPLACE_DATA);
		
		for(int icv=0; icv<ncv; icv++) {
			aug.calculate(icv);
			betaAugVec[0][icv] = aug.beta;
			// cout << "\t" << "enthalpy and grad: " << enthalpy[icv] << ", " << grad_enthalpy[icv][0] << ", " << grad_enthalpy[icv][1] << ", " << grad_enthalpy[icv][2] << endl;
			// betaAugVec[0][icv] = 0.0;
		}
		updateCvDataG1G2(betaAugVec[0], REPLACE_DATA);		
		
		// PRINT("Doing Variable PrTurb calculation \n")
		int icv0, icv1; double w0, w1;
		for(int ifa=nfa_b; ifa<nfa_b2gg; ifa++) {
			if(ifa<nfa || (ifa>=nfa_b2 && ifa<nfa_b2gg)) {
				getInterpolationFactors(ifa, icv0, icv1, w0, w1);
				
				// PrTurb_fa[ifa] = (w1*beta[icv0] + w0*beta[icv1])*PrTurb;
				// PrTurb_fa[ifa] = (w1*betaAugVec[0][icv0] + w0*betaAugVec[0][icv1])*PrTurb;
				PrTurb_fa[ifa] = exp(w1*betaAugVec[0][icv0] + w0*betaAugVec[0][icv1])*PrTurb;
			}
		}

		for(list<FaZone>::iterator ZZ=faZoneList.begin(); ZZ!=faZoneList.end(); ZZ++) {
			if(ZZ->getKind() == FA_ZONE_BOUNDARY) {
				if(zoneIsWall(ZZ->getName())) {
        			for(auto & ifa: ZZ->faVec) {
						icv0 = cvofa[ifa][0];
						// PrTurb_fa[ifa] = beta[icv0]*PrTurb;
						// PrTurb_fa[ifa] = betaAugVec[0][icv0]*PrTurb;
						PrTurb_fa[ifa] = exp(betaAugVec[0][icv0])*PrTurb;
					}
				}

				if(!zoneIsWall(ZZ->getName())) {
					for(auto & ifa: ZZ->faVec) {
						icv0 = cvofa[ifa][0];
						icv1 = cvofa[ifa][1];
						// PrTurb_fa[ifa] = beta[icv1]*PrTurb;
						// PrTurb_fa[ifa] = beta[icv0]*PrTurb; // <-- Use this for now so I don't get NaNs
						// PrTurb_fa[ifa] = betaAugVec[0][icv0]*PrTurb; // <-- Use this for now so I don't get NaNs
						PrTurb_fa[ifa] = exp(betaAugVec[0][icv0])*PrTurb; // <-- Use this for now so I don't get NaNs
					}
				}
			}
		}
	}

	void diffusivityHookScalarRansTurb(const std::string &name) {
		ScalarTranspEq *eq;
		double sigma;

		eq = getScalarTransportData(name);

		double rho_fa, kine_fa, omega_fa, temp_stag_fa;
		int icv0, icv1; double w0, w1;
		
		if (name == "kine" || name == "omega") {
			if (name=="kine") {sigma = sigma_k;}
			else if (name=="omega") {sigma = sigma_om;}

			for (int ifa=nfa_b; ifa<nfa; ifa++) {
				getInterpolationFactors(ifa, icv0, icv1, w0, w1);
				rho_fa = w1*rho[icv0] + w0*rho[icv1];
				kine_fa = w1*kine[icv0] + w0*kine[icv1];
				omega_fa = w1*omega[icv0] + w0*omega[icv1];
				
				eq->diff[ifa] = mul_fa[ifa] + sigma*rho_fa*kine_fa/omega_fa;
			}

			for(list<FaZone>::iterator ZZ=faZoneList.begin(); ZZ!=faZoneList.end(); ZZ++) {
				if(ZZ->getKind() == FA_ZONE_BOUNDARY) {
					if(zoneIsWall(ZZ->getName())) {
						for(int ifa=ZZ->ifa_f; ifa<=ZZ->ifa_l; ifa++) {
							int icv0 = cvofa[ifa][0];
							eq->diff[ifa] = mul_fa[ifa];
						}
					}
					if(!zoneIsWall(ZZ->getName())) {
						for(int ifa=ZZ->ifa_f; ifa<=ZZ->ifa_l; ifa++) {
							int icv1 = cvofa[ifa][1];
							eq->diff[ifa] = mul_fa[ifa] + sigma*rho[icv1]*kine[icv1]/omega[icv1];
						}
					}
				}
			}
		}

		else if (name == "psi_rs") {
			// sigma = 1.0;
			// for (int ifa=nfa_b; ifa<nfa; ifa++) eq->diff[ifa] = 0.0;
			for (int ifa=nfa_b; ifa<nfa; ifa++) {
				getInterpolationFactors(ifa, icv0, icv1, w0, w1);
				// rho_fa = w1*rho[icv0] + w0*rho[icv1];
				// kine_fa = w1*kine[icv0] + w0*kine[icv1];
				// omega_fa = w1*omega[icv0] + w0*omega[icv1];
				
				eq->diff[ifa] = 0; //ls_a*(mul_fa[ifa] + sigma*rho_fa*kine_fa/omega_fa);
			}

			for(list<FaZone>::iterator ZZ=faZoneList.begin(); ZZ!=faZoneList.end(); ZZ++) {
				if(ZZ->getKind() == FA_ZONE_BOUNDARY) {
					if(zoneIsWall(ZZ->getName())) {
						for(int ifa=ZZ->ifa_f; ifa<=ZZ->ifa_l; ifa++) {
							int icv0 = cvofa[ifa][0];

							eq->diff[ifa] = 0; //ls_a*mul_fa[ifa];
						}
					}
					if(!zoneIsWall(ZZ->getName())) {
						for(int ifa=ZZ->ifa_f; ifa<=ZZ->ifa_l; ifa++) {
							int icv1 = cvofa[ifa][1];
							
							eq->diff[ifa] = 0; //ls_a*(mul_fa[ifa] + sigma*rho[icv1]*kine[icv1]/omega[icv1]);
						}
					}
					// for(int ifa=ZZ->ifa_f; ifa<=ZZ->ifa_l; ifa++) eq->diff[ifa] = 0.0;
				}
			}
		}
	}

	void boundaryHookScalarRansTurb(double *phi, FaZone *zone, const std::string &name) {
		// Enforce the wall BC for omega
		if (name=="omega") {
			if (zone->getKind()==FA_ZONE_BOUNDARY) {
				if (zoneIsWall(zone->getName())) {
					for(int index=0; index<zone->faVec.size(); index++) {
						int ifa = zone->faVec[index];
						int icv0 = cvofa[ifa][0];
						int icv1 = cvofa[ifa][1];
						phi[icv1] = 6.0*calcMuLam(icv1)/(rho[icv0]*beta_0*wallDist[icv0]*wallDist[icv0]);
					}
				}
			}
		}

		// Enforce the custom inflow BC
		if (zone->getKind()==FA_ZONE_BOUNDARY) {
			// printf("Checking if boundary is inflow\n");
			if (zone->getNameString()=="inlet" || zone->getNameString()=="inflow") {
				int iFace, lCell, rCell, pos, dir=1; // y-normal so dir=1
				double area, nVec[3], intfac;
				// printf("Zone size is: %d", zone->faVec.size());
				// DEBUG_PRINT(0, "Starting loop in boundaryHookScalarRansTurb\n");
				for (int iFaceIndex=0; iFaceIndex < zone->faVec.size(); iFaceIndex++) {
					// DEBUG_PRINT(0, "On iFaceIndex=%d \n", iFaceIndex);
					iFace = zone->faVec[iFaceIndex]; lCell = cvofa[iFace][0]; rCell = cvofa[iFace][1]; pos=0;
					area  = normVec3d(nVec, fa_normal[iFace]);
					while(inletData[pos].x[dir] < x_fa[iFace][dir]) {
						pos++; 
						if(pos >= inletData.size()){
							fprintf(stderr, "\n\nERROR: PROVIDED INPUT DATA "
											"INSUFFICIENT FOR INTERPOLATION\n\n");
							throw(-1);
						}
					}
					// DEBUG_PRINT(0, "%.10le, %.10le\n", inletData[pos].x[dir], x_fa[iFace][dir])
					if(pos == 0){
						fprintf(stderr, "\n\nERROR: ENSURE THAT THE INLET DATA BEGINS "
										"FROM THE Y-COORDINATE SET TO ZERO\n\n");
						throw(-1);
					}
					intfac = (x_fa[iFace][dir]-inletData[pos-1].x[dir]) / (inletData[pos].x[dir]-inletData[pos-1].x[dir]);
					if (name=="kine") {
						// printf("Updating kine");
						phi[rCell] = (1-intfac) * inletData[pos-1].kine + intfac * inletData[pos].kine;
						// assert(phi[rCell] == phi[rCell]);
					}
					if (name=="omega") {
						// printf("Updating omega");
						phi[rCell] = (1-intfac) * inletData[pos-1].omega + intfac * inletData[pos].omega;
						// assert(phi[rCell] == phi[rCell]);
						// fprintf(stdout, "We changed nuSA at boundary to nuSA = %le", phi[rCell]);
					}
				}
				// DEBUG_PRINT(0, "Finished loop in boundaryHookScalarRansTurb\n");
			}
		}
	}

	void sourceHookRansTurbCoupled(double **rhs, double ***A, int nScal, int flagImplicit) {
		int noc00, index;
		double src, nuT, Prod, crossProd, f_beta, beta_full;
		double velmag, len_scale, dil_factor;
		double d_src;

		for(int icv=0; icv<ncv; icv++) {
			noc00 = nbocv_i[icv];

			muLam[icv] = calcMuLam(icv);
			nuT = muT[icv]/rho[icv];			// Better to calculate again, which I do in calcviscMuet

			if (use_mod_prod) Prod = nuT*strMag[icv]*strMag[icv];						// _m version
			else Prod = nuT*strMag[icv]*strMag[icv] - 2.0/3.0*diverg[icv]*kine[icv];	// Standard version
			
			if (use_klim) Prod = fmin(Prod, 20*beta_star*omega[icv]*kine[icv]);			// _klim version
			
			crossProd = 0.0;
			for(int i=0; i<3; i++) crossProd += grad_kine[icv][i] * grad_omega[icv][i];
			crossProd *= calc_sigma_d(crossProd)/omega[icv];

			// kine :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
			index = getScalarTransportIndex("kine");
			src = Prod - beta_star*omega[icv]*kine[icv];
			d_src = -beta_star*omega[icv];

			rhs[icv][5+index] += rho[icv] * src * cv_volume[icv];
        	if(flagImplicit) A[noc00][5+index][5+index] -= d_src * cv_volume[icv];

			// omega ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
			index = getScalarTransportIndex("omega");
			f_beta = 1.0;
			beta_full = calc_beta_full(f_beta);
			src = gamma_kom*omega[icv]*Prod/kine[icv] - beta_full*omega[icv]*omega[icv] + crossProd;
			// src = betaAugVec[0][icv]*gamma_kom*omega[icv]*Prod/kine[icv] - beta_full*omega[icv]*omega[icv] + crossProd;
			d_src = -2*beta_full*omega[icv]; // - crossProd/omega[icv];

			rhs[icv][5+index] += rho[icv] * src * cv_volume[icv];
        	if(flagImplicit) A[noc00][5+index][5+index] -= d_src * cv_volume[icv];

			// psi_rs :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
			index = getScalarTransportIndex("psi_rs");
			velmag = sqrt(vel[icv][0]*vel[icv][0] + vel[icv][1]*vel[icv][1] + vel[icv][2]*vel[icv][2]);
			len_scale = ls_a*sqrt(kine[icv])/omega[icv];
			// dil_factor = velmag/sqrt(gamma[icv]*press[icv]/rho[icv]);
			src = psi_rs[icv]*diverg[icv] - (psi_rs[icv]-psi_rs_ref)*velmag/len_scale;
			d_src = -velmag/len_scale;

			rhs[icv][5+index] += rho[icv] * src * cv_volume[icv];
        	if(flagImplicit) A[noc00][5+index][5+index] -= d_src * cv_volume[icv];
		}
	}
};

#endif