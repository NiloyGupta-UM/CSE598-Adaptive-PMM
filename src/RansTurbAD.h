#ifndef RANSTURB_AD_H
#define RANSTURB_AD_H

#include "RansTurb.h"
#include "UgpWithCvCompFlowAD.h"

class RansTurb_AD : virtual public UgpWithCvCompFlow_AD, public RansTurb
{
	public:

		adouble *muT_AD, *marker_AD;
		adouble (*grad_kine)[3], *kine_diff;
		double *kine_adjoint, *dR_kine__dbeta;
		adouble *omega, (*grad_omega)[3], *omega_diff;
		double *omega_adjoint, *dR_omega__dbeta;
		adouble *psi_rs, (*grad_psi_rs)[3], *psi_rs_diff;
		double *psi_rs_adjoint, *dR_psi_rs__dbeta;
	
	public:

		RansTurb_AD(void) {
			if(mpi_rank==0) std::cout<<"RansTurb_AD()"<<std::endl;
			kine_adjoint = NULL; registerScalar(kine_adjoint, "kine_adjoint", CV_DATA);
			dR_kine__dbeta = NULL; registerScalar(dR_kine__dbeta, "dR_kine__dbeta", CV_DATA);
			omega_adjoint = NULL; registerScalar(omega_adjoint, "omega_adjoint", CV_DATA);
			dR_omega__dbeta = NULL; registerScalar(dR_omega__dbeta, "dR_omega__dbeta", CV_DATA);
			psi_rs_adjoint = NULL; registerScalar(psi_rs_adjoint, "psi_rs_adjoint", CV_DATA);
			dR_psi_rs__dbeta = NULL; registerScalar(dR_psi_rs__dbeta, "dR_psi_rs__dbeta", CV_DATA);

			augVars.rho_AD = &rho_AD;
			augVars.rhou_AD = &rhou_AD;
			augVars.rhoE_AD = &rhoE_AD;
			augVars.diverg_AD = &diverg;
			augVars.strMag_AD = &strMag;
			augVars.vortMag_AD = &vortMag;
			augVars.mu_AD = &muLam_AD;
			augVars.temperature_AD = &temp;
			augVars.grad_enthalpy_AD = &grad_enthalpy;
			augVars.marker_AD = &marker_AD;
			// cout << "pointers test: " << augVars.grad_enthalpy_AD << ", " << &grad_enthalpy << endl;
			augVars.RoM_AD = &RoM;
			augVars.scal_AD.resize(3);
			augVars.grad_scal_AD.resize(3);
			augVars.scal_AD[0] = &kine;
			augVars.grad_scal_AD[0] = &grad_kine;
			augVars.scal_AD[1] = &omega;
			augVars.grad_scal_AD[1] = &grad_omega;
			augVars.scal_AD[2] = &psi_rs;
			augVars.grad_scal_AD[2] = &grad_psi_rs;

			#ifdef ADJOINT_MODE
				MPI_Barrier(mpi_comm);
				aug.init(&augVars);
			#endif

			aug.psi_kine = NULL; registerScalar(aug.psi_kine, "aug.psi_kine", CV_DATA);
			aug.psi_omega = NULL; registerScalar(aug.psi_omega, "aug.psi_omega", CV_DATA);
			aug.psi_psi_rs = NULL; registerScalar(aug.psi_psi_rs, "aug.psi_psi_rs", CV_DATA);
		}

		void copy_turb_adjoint(void) {
			int nScal = scalarTranspEqVector.size();
			for(int i=0; i<nScal; i++)
			{
				if(strcmp(scalarTranspEqVector_AD[i].name,"kine")==0) {
					for(int icv=0; icv<ncv_gg; icv++) {
						kine_adjoint[icv] = scalarTranspEqVector_psi[i].phi[icv];
					}
				}
				if(strcmp(scalarTranspEqVector_AD[i].name,"omega")==0) {
					for(int icv=0; icv<ncv_gg; icv++) {
						omega_adjoint[icv] = scalarTranspEqVector_psi[i].phi[icv];
					}
				}
				if(strcmp(scalarTranspEqVector_AD[i].name,"psi_rs")==0) {
					for(int icv=0; icv<ncv_gg; icv++) {
						psi_rs_adjoint[icv] = scalarTranspEqVector_psi[i].phi[icv];
					}
				}
			}
		}

		void initialHookScalarRansTurbModel_AD(void) {
			if(mpi_rank==0) printf("initialHookScalarRansTurbModel_AD()\n");
			RansTurb::initialHookScalarRansTurbModel();
			connectPointers("kine", &kine, &grad_kine, &kine_diff);
			connectPointers("omega", &omega, &grad_omega, &omega_diff);
			connectPointers("psi_rs", &psi_rs, &grad_psi_rs, &psi_rs_diff);
			updateCvDataG1G2(wallDist, REPLACE_DATA);

			marker_AD = new adouble[ncv_gg];

			// Now we need to fill in values for betaAug_AD, using dummy variable
			// for (int icv=0; icv<ncv_gg; icv++) betaAugVec_AD[0][icv].setValue(betaAug[icv]);
			PRINT("initialHookScalarRansTurbModel_AD() Done\n");
		}

		void connectPointers(const std::string &name, adouble **q, adouble (**grad_q)[3], adouble **q_diff) {
			int nScal = scalarTranspEqVector.size();
			for(int i=0; i<nScal; i++) {
				if(strcmp(scalarTranspEqVector_AD[i].name, name.c_str())==0) {
					(*q)      = scalarTranspEqVector_AD[i].phi;
					(*grad_q) = scalarTranspEqVector_AD[i].grad_phi;
					(*q_diff) = scalarTranspEqVector_AD[i].diff;
				}
			}
		}

		void initialize_turb_adjoint(void) { readAdjointsFromFile(); }

		void writeAdjointsToFile() {
			int ncv_global;
			MPI_Allreduce(&ncv, &ncv_global, 1, MPI_INT, MPI_SUM, mpi_comm);
			PRINT("Writing adjoints to file \n");
			PRINT("\tTotal number of cv's across all processors = %d\n", ncv_global);
			int nScal = scalarTranspEqVector.size();
			double *psi__kine, *psi__omega, *psi__psi_rs;
			for(int i=0; i<nScal; i++) {
				if(strcmp(scalarTranspEqVector_AD[i].name, "kine")==0) {
					psi__kine   = scalarTranspEqVector_psi[i].phi; PRINT("\tkine pointer connected\n");
				}
				if(strcmp(scalarTranspEqVector_AD[i].name, "omega")==0) {
					psi__omega   = scalarTranspEqVector_psi[i].phi; PRINT("\tomega pointer connected\n");
				}
				if(strcmp(scalarTranspEqVector_AD[i].name, "psi_rs")==0) {
					psi__psi_rs  = scalarTranspEqVector_psi[i].phi; PRINT("\tpsi_rs pointer connected\n");
				}
			}
			for (int proc=0; proc<mpi_size; proc++) {
				if (proc==mpi_rank) {
					FILE *fp;
					if(mpi_rank==0) fp = fopen("adjoint_values.txt", "w");
					else            fp = fopen("adjoint_values.txt", "a");
					for(int icv=0; icv<ncv; icv++)
					{
						fprintf(fp, "%09ld  %+.15le  %+.15le  %+.15le  %+.15le  %+.15le  %+.15le  %+.15le  %+.15le  \n",
							std::lround(global_id[icv]),
							psi_rho[icv], psi_rhou[icv][0], psi_rhou[icv][1], psi_rhou[icv][2], psi_rhoE[icv],
							psi__kine[icv], psi__omega[icv], psi__psi_rs[icv]);
					}
					fclose(fp);
				}
				MPI_Barrier(mpi_comm);
    		}	
		}

		void readAdjointsFromFile() {
			PRINT("\n\nReading adjoints values from text file, if provided\nCreating local_id vector\n");

			int ncv_global;
			PRINT("Number of cv's on this processor = %d\n", ncv);
			MPI_Allreduce(&ncv, &ncv_global, 1, MPI_INT, MPI_SUM, mpi_comm);
			PRINT("Total number of cv's across processors = %d\n", ncv_global);
			std::vector<int> local_id(ncv_global, -1);
			for(int icv=0; icv<ncv; icv++)
				local_id[std::lround(global_id[icv])] = icv;

			int nScal = scalarTranspEqVector.size();
			double *psi__kine, *psi__omega, *psi__psi_rs;
			for(int i=0; i<nScal; i++)
			{
				if(strcmp(scalarTranspEqVector_AD[i].name, "kine")==0) {
					psi__kine   = scalarTranspEqVector_psi[i].phi; PRINT("\tkine pointer connected\n");
				}
				if(strcmp(scalarTranspEqVector_AD[i].name, "omega")==0) {
					psi__omega   = scalarTranspEqVector_psi[i].phi; PRINT("\tomega pointer connected\n");
				}
				if(strcmp(scalarTranspEqVector_AD[i].name, "psi_rs")==0) {
					psi__psi_rs  = scalarTranspEqVector_psi[i].phi; PRINT("\tpsi_rs pointer connected\n");
				}
			}

			PRINT("Checking if file exists and reading...\n");
			
			for (int proc=0; proc<mpi_size; proc++) {
				if (proc==mpi_rank) {
					printf("Reading on processor %d\n", mpi_rank);
					FILE *fp = fopen("adjoint_values.txt", "r");
					if(fp!=NULL)
					{
						int rtnval, gloId, locId;
						double temp;
						printf("\tReading line 000000000");
						for(int iLine=0; iLine<ncv_global; iLine++)
						{
							printf("\b\b\b\b\b\b\b\b\b%09d", iLine+1);
							rtnval = fscanf(fp, "%d", &gloId); locId = local_id[gloId];
							rtnval = fscanf(fp, "%le", &temp); if(locId>=0) psi_rho[locId]     = temp;
							rtnval = fscanf(fp, "%le", &temp); if(locId>=0) psi_rhou[locId][0] = temp;
							rtnval = fscanf(fp, "%le", &temp); if(locId>=0) psi_rhou[locId][1] = temp;
							rtnval = fscanf(fp, "%le", &temp); if(locId>=0) psi_rhou[locId][2] = temp;
							rtnval = fscanf(fp, "%le", &temp); if(locId>=0) psi_rhoE[locId]    = temp;
							rtnval = fscanf(fp, "%le", &temp); if(locId>=0) psi__kine[locId]   = temp;
							rtnval = fscanf(fp, "%le", &temp); if(locId>=0) psi__omega[locId]   = temp;
							rtnval = fscanf(fp, "%le", &temp); if(locId>=0) psi__psi_rs[locId]   = temp;
						}
						printf("\n");
						fclose(fp);
					}
				}
				MPI_Barrier(mpi_comm);
			}
			
			updateCvDataG1G2(psi_rho, REPLACE_DATA);
			updateCvDataG1G2(psi_rhou, REPLACE_ROTATE_DATA);
			updateCvDataG1G2(psi_rhoE, REPLACE_DATA);
			updateCvDataG1G2(psi__kine, REPLACE_DATA);
			updateCvDataG1G2(psi__omega, REPLACE_DATA);
			updateCvDataG1G2(psi__psi_rs, REPLACE_DATA);
			PRINT("\n\n\n");
		}

		double calc_gamma_kom_AD(void) {return beta_0/beta_star - sigma_om*kappa_2/sqrt_beta_star;} 
		adouble calc_sigma_d_AD(adouble crossProd) {return (crossProd > 0) ? 0.125 : 0;}
		double calc_beta_full_AD(double f_beta) {return beta_0*f_beta;}

		void calcMarkerFunction_AD(void) {
			adouble s_vec[3], g_vec[3], vel_mag, g_mag, f_mark;

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
				marker_AD[icv] = f_mark * kine[icv] / (vel_mag * vel_mag);
			}
		}

		void calcRansTurbViscMuet_AD(adouble *rho, adouble (*rhou)[3]) {
			calcStrainRateAndDivergence_AD(); calcVorticity_AD();
			calcMarkerFunction_AD();			

			adouble rho_fa, kine_fa, omega_fa, omega_hat, strMag_fa;
			int icv0, icv1; double w0, w1;

			for(int ifa=nfa_b; ifa<nfa_b2gg; ifa++) {
				if(ifa<nfa || (ifa>=nfa_b2 && ifa<nfa_b2gg)) {
					getInterpolationFactors(ifa, icv0, icv1, w0, w1);

					rho_fa = w1*rho_AD[icv0] + w0*rho_AD[icv1];
					kine_fa = w1*kine[icv0] + w0*kine[icv1];
					omega_fa = w1*omega[icv0] + w0*omega[icv1];
					strMag_fa = w1*strMag[icv0] + w0*strMag[icv1];
					omega_hat = fmax(omega_fa, C_lim*strMag_fa/sqrt_beta_star);
					
					mut_fa[ifa] = rho_fa*kine_fa/omega_hat;
				}
			}

			for(list<FaZone>::iterator ZZ=faZoneList.begin(); ZZ!=faZoneList.end(); ZZ++) {
				if(ZZ->getKind() == FA_ZONE_BOUNDARY) {
					if(zoneIsWall(ZZ->getName())) {
						for(int index=0; index<ZZ->faVec.size(); index++) {
							int ifa = ZZ->faVec[index];
          					mut_fa[ifa] = 0.0;        
						}
					}
					if(!zoneIsWall(ZZ->getName())) {
						for(int index=0; index<ZZ->faVec.size(); index++) {
							int ifa = ZZ->faVec[index];
							icv0 = cvofa[ifa][0];
							icv1 = cvofa[ifa][1];
							omega_hat = fmax(omega[icv1], C_lim*strMag[icv0]/sqrt_beta_star);
							mut_fa[ifa] = rho_AD[icv1]*kine[icv1]/omega_hat;
						}
					}
				}
			}
		}

		void calcTurbulentPrandtlNumber_AD() {
			// Some prep stuff for LIFE that we may not actually need????
			// updateCvDataG1G2(grad_enthalpy, REPLACE_ROTATE_DATA);		// <-- not needed here as values should be updated?
			// calcCv2Grad_AD(grad_enthalpy, enthalpy, limiterNavierS, enthalpy, epsilonSDWLS);       // enthalpy gradients
			// // Might get seg fault for doing icv until ncv_gg instead of ncv_g
			// for(int icv=0; icv<ncv_gg; icv++) {
			// 	aug.calculate_AD(icv);
			// 	betaAugVec_AD[0][icv] = aug.beta_AD;
			// }
			
			
			int icv0, icv1; double w0, w1;
			for(int ifa=nfa_b; ifa<nfa_b2gg; ifa++) {
				if(ifa<nfa || (ifa>=nfa_b2 && ifa<nfa_b2gg)) {
					getInterpolationFactors(ifa, icv0, icv1, w0, w1);
					
					// PrTurb_fa[ifa] = (w1*beta[icv0] + w0*beta[icv1])*PrTurb;
					// PrTurb_fa[ifa] = (w1*betaAugVec_AD[0][icv0] + w0*betaAugVec_AD[0][icv1])*PrTurb;
					PrTurb_fa[ifa] = exp(w1*betaAugVec_AD[0][icv0] + w0*betaAugVec_AD[0][icv1])*PrTurb;
				}
			}

			for(list<FaZone>::iterator ZZ=faZoneList.begin(); ZZ!=faZoneList.end(); ZZ++) {
				if(ZZ->getKind() == FA_ZONE_BOUNDARY) {
					if(zoneIsWall(ZZ->getName())) {
						for(int index=0; index<ZZ->faVec.size(); index++) {
							int ifa = ZZ->faVec[index];
          					icv0 = cvofa[ifa][0];
							// PrTurb_fa[ifa] = beta[icv0]*PrTurb;
							// PrTurb_fa[ifa] = betaAugVec_AD[0][icv0]*PrTurb;
							PrTurb_fa[ifa] = exp(betaAugVec_AD[0][icv0])*PrTurb;
						}
					}
					if(!zoneIsWall(ZZ->getName())) {
						for(int index=0; index<ZZ->faVec.size(); index++) {
							int ifa = ZZ->faVec[index];
							icv0 = cvofa[ifa][0];
							icv1 = cvofa[ifa][1];
							// PrTurb_fa[ifa] = beta[icv1]*PrTurb;
							// PrTurb_fa[ifa] = beta[icv0]*PrTurb; // <-- Use this for now so I don't get NaNs
							// PrTurb_fa[ifa] = betaAugVec_AD[0][icv0]*PrTurb; // <-- Use this for now so I don't get NaNs
							PrTurb_fa[ifa] = exp(betaAugVec_AD[0][icv0])*PrTurb; // <-- Use this for now so I don't get NaNs
						}
					}
				}
			}
		}

		void diffusivityHookScalarRansTurb_AD(const std::string &name) {
			ScalarTranspEq *eq; adouble *diff;
			double sigma;

			adouble rho_fa, kine_fa, omega_fa;
			int icv0, icv1; double w0, w1;
			
			if (name == "kine" || name == "omega") {
				if (name=="kine") {sigma = sigma_k; diff = kine_diff;}
				else if (name=="omega") {sigma = sigma_om; diff = omega_diff;}

				for (int ifa=nfa_b; ifa<nfa_b2gg; ifa++) {
					if(ifa<nfa || (ifa>=nfa_b2 && ifa<nfa_b2gg)) {	
						getInterpolationFactors(ifa, icv0, icv1, w0, w1);
						rho_fa = w1*rho_AD[icv0] + w0*rho_AD[icv1];
						kine_fa = w1*kine[icv0] + w0*kine[icv1];
						omega_fa = w1*omega[icv0] + w0*omega[icv1];
						
						diff[ifa] = mul_fa[ifa] + sigma*rho_fa*kine_fa/omega_fa;
					}
				}

				for(list<FaZone>::iterator ZZ=faZoneList.begin(); ZZ!=faZoneList.end(); ZZ++) {
					if(ZZ->getKind() == FA_ZONE_BOUNDARY) {
						if(zoneIsWall(ZZ->getName())) {
							for(int index=0; index<ZZ->faVec.size(); index++) {
								int ifa = ZZ->faVec[index];
								icv0 = cvofa[ifa][0];
								diff[ifa] = mul_fa[ifa];
							}
						}
						if(!zoneIsWall(ZZ->getName())) {
							for(int index=0; index<ZZ->faVec.size(); index++) {
								int ifa = ZZ->faVec[index];
								icv1 = cvofa[ifa][1];
								diff[ifa] = mul_fa[ifa] + sigma*rho_AD[icv1]*kine[icv1]/omega[icv1];
							}
						}
					}
				}
			}

			else if (name == "psi_rs") {
				diff = psi_rs_diff;

				for (int ifa=nfa_b; ifa<nfa_b2gg; ifa++) {
					if(ifa<nfa || (ifa>=nfa_b2 && ifa<nfa_b2gg)) diff[ifa] = 0; //ls_a*mul_fa[ifa];
				}

				for(list<FaZone>::iterator ZZ=faZoneList.begin(); ZZ!=faZoneList.end(); ZZ++) {
					if(ZZ->getKind() == FA_ZONE_BOUNDARY) {
					// 	if(zoneIsWall(ZZ->getName())) {
					// 		for(int index=0; index<ZZ->faVec.size(); index++) {
					// 			int ifa = ZZ->faVec[index];
					// 			icv0 = cvofa[ifa][0];
					// 			diff[ifa] = mul_fa[ifa];
					// 		}
					// 	}
					// 	if(!zoneIsWall(ZZ->getName())) {
					// 		for(int index=0; index<ZZ->faVec.size(); index++) {
					// 			int ifa = ZZ->faVec[index];
					// 			icv1 = cvofa[ifa][1];
					// 			diff[ifa] = mul_fa[ifa] + sigma*rho_AD[icv1]*kine[icv1]/omega[icv1];
					// 		}
					// 	}
						for(int index=0; index<ZZ->faVec.size(); index++) {
							int ifa = ZZ->faVec[index];
							diff[ifa] = 0; //ls_a*mul_fa[ifa];
						}
					}

				}
			}
		}

		void boundaryHookScalarRansTurb_AD(adouble *phi, FaZone *zone, const std::string &name) {
			// Enforce the wall BC for omega
			if (name=="omega") {
				if (zone->getKind()==FA_ZONE_BOUNDARY) {
					if (zoneIsWall(zone->getName())) {
						for(int index=0; index<zone->faVec.size(); index++) {
							int ifa = zone->faVec[index];
							int icv0 = cvofa[ifa][0];
							int icv1 = cvofa[ifa][1];
							phi[icv1] = 6.0*calcMuLam_AD(icv1)/(rho_AD[icv0]*beta_0*wallDist[icv0]*wallDist[icv0]);
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

		virtual void sourceHookRansTurbCoupled_AD(adouble **rhs, double ***A, int flagImplicit) {
			DEBUG_PRINT(0, "Started sourceHookRansTurbCoupled_AD in RansTurbAD.h\n")
			int noc00, index;
			adouble src, nuT, Prod, crossProd;
			adouble velmag, len_scale;
			double f_beta, beta_full; // Note beta_full is distinct from beta field
			double d_src;
			// DEBUG_PRINT(0, "Entered for loop over icv's\n")
			// for (int proc=0; proc<mpi_size; proc++) {
			// 	if (proc==mpi_rank)
			for (int icv=0; icv<ncv_g; icv++) {
				// cout << "\t" << "enthalpy and grad: " << enthalpy[icv].value() << ", " << grad_enthalpy[icv][0].value() << ", " << grad_enthalpy[icv][1].value() << ", " << grad_enthalpy[icv][2].value() << endl;
				// DEBUG_PRINT(proc, "At icv %d on proc %d ncv=%d, ncv_g=%d \n", icv, proc, ncv, ncv_g)
				noc00 = nbocv_i[icv];

				muLam_AD[icv] = calcMuLam_AD(icv);
				nuT = kine[icv]/fmax(omega[icv], C_lim*strMag[icv]/sqrt_beta_star);			// Better to calculate again, which I do in calcviscMuet
				// DEBUG_PRINT(proc, "Successfully calculated viscosities\n")

				if (use_mod_prod) Prod = nuT*strMag[icv]*strMag[icv];						// _m version
				else Prod = nuT*strMag[icv]*strMag[icv] - 2.0/3.0*diverg[icv]*kine[icv];	// Standard version
				
				if (use_klim) Prod = fmin(Prod, 20*beta_star*omega[icv]*kine[icv]);			// _klim version
				// DEBUG_PRINT(proc, "Successfully calculated production\n")
				
				crossProd = 0.0;
				for(int i=0; i<3; i++) crossProd += grad_kine[icv][i] * grad_omega[icv][i];
				crossProd *= calc_sigma_d_AD(crossProd)/omega[icv];
				// DEBUG_PRINT(proc, "Successfully calculated crossproduction\n")

				// kine :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
				index = getScalarTransportIndex("kine");
				// DEBUG_PRINT(proc, "beta: %.8le \n", beta[icv]);
				// DEBUG_PRINT(proc, "omega: %.8le \n", omega[icv].value());
				// DEBUG_PRINT(proc, "kine: %.8le \n", kine[icv].value());
				src = Prod - beta_star*omega[icv]*kine[icv];
				// DEBUG_PRINT(proc, "src: %.8le \n", src.value());
				d_src = -beta_star*omega[icv].value();

				// DEBUG_PRINT(proc, "rho: %.8le \n", rho_AD[icv].value());
				// DEBUG_PRINT(proc, "cvvol: %.8le \n", cv_volume[icv]);
				// dR_kine__dbeta[icv] = Prod.value()*rho_AD[icv].value()*cv_volume[icv];
				// dR_kine__dbeta[icv] = 0.0;

				rhs[icv][5+index] += rho_AD[icv] * src * cv_volume[icv];
				if(flagImplicit && icv<ncv) A[noc00][5+index][5+index] -= d_src * cv_volume[icv];
				// DEBUG_PRINT(proc, "Successfully calculated kine part\n")

				// omega ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
				index = getScalarTransportIndex("omega");
				f_beta = 1.0;
				beta_full = calc_beta_full_AD(f_beta);
				src = gamma_kom*omega[icv]*Prod/kine[icv] - beta_full*omega[icv]*omega[icv] + crossProd;
				// src = betaAugVec_AD[0][icv]*gamma_kom*omega[icv]*Prod/kine[icv] - beta_full*omega[icv]*omega[icv] + crossProd;
				d_src = -2*beta_full*omega[icv].value(); // - crossProd/omega[icv];

				// dR_omega__dbeta[icv] = gamma_kom*omega[icv].value()*Prod.value()/kine[icv].value()*rho_AD[icv].value()*cv_volume[icv];

				rhs[icv][5+index] += rho_AD[icv] * src * cv_volume[icv];
				if(flagImplicit && icv<ncv) A[noc00][5+index][5+index] -= d_src * cv_volume[icv];
				// DEBUG_PRINT(proc, "Successfully calculated omega part\n")

				// psi_rs :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
				index = getScalarTransportIndex("psi_rs");
				velmag = sqrt(vel[icv][0]*vel[icv][0] + vel[icv][1]*vel[icv][1] + vel[icv][2]*vel[icv][2]);
				// DEBUG_PRINT(proc, "velmag: %.8le \n", velmag.value());
				len_scale = ls_a*sqrt(kine[icv])/omega[icv];
				// DEBUG_PRINT(proc, "len_scale: %.8le \n", len_scale.value());
				src = psi_rs[icv]*diverg[icv] - (psi_rs[icv]-psi_rs_ref)*velmag/len_scale;
				// DEBUG_PRINT(proc, "src: %.8le \n", src.value());
				d_src = -velmag.value()/len_scale.value();

				rhs[icv][5+index] += rho_AD[icv] * src * cv_volume[icv];
				if(flagImplicit && icv<ncv) A[noc00][5+index][5+index] -= d_src * cv_volume[icv];
				// DEBUG_PRINT(proc, "Successfully calculated psi_rs part\n")
			}
			// 	MPI_Barrier(mpi_comm);
			// }
			DEBUG_PRINT(0, "Finished sourceHookRansTurbCoupled_AD in RansTurbAD.h\n")
		}
		
		void writeSens(void) {
			int nScal = scalarTranspEqVector.size();
			double *psi__kine, *psi__omega, *psi__psi_rs;
			for(int i=0; i<nScal; i++) {
				if(strcmp(scalarTranspEqVector_AD[i].name, "kine")==0) {
					psi__kine   = scalarTranspEqVector_psi[i].phi; PRINT("\tkine pointer connected\n");
				}
				if(strcmp(scalarTranspEqVector_AD[i].name, "omega")==0) {
					psi__omega   = scalarTranspEqVector_psi[i].phi; PRINT("\tomega pointer connected\n");
				}
				if(strcmp(scalarTranspEqVector_AD[i].name, "psi_rs]")==0) {
					psi__psi_rs  = scalarTranspEqVector_psi[i].phi; PRINT("\tpsi_rs pointer connected\n");
				}
			}
			for (int proc=0; proc<mpi_size; proc++) {
				if (proc==mpi_rank) {
					FILE *fp;
					if(mpi_rank==0) fp = fopen("beta_sens.dat", "w");
					else            fp = fopen("beta_sens.dat", "a");
					for(int icv=0; icv<ncv; icv++)
					{
						fprintf(fp, "%09ld  %+.15le  %+.15le  %+.15le \n",
							std::lround(global_id[icv]), dR_kine__dbeta[icv]*psi__kine[icv], dR_omega__dbeta[icv]*psi__omega[icv], dR_psi_rs__dbeta[icv]*psi__psi_rs[icv]);
					}
					fclose(fp);
				}
				MPI_Barrier(mpi_comm);
			}
    	}
};

#endif