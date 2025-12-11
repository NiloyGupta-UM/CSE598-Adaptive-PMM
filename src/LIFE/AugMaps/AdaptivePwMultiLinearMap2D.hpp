#ifndef AdaptivePwMultiLinearMap2D_hpp
#define AdaptivePwMultiLinearMap2D_hpp

#include <cstdio>
#include "GenAugMap.hpp"

// For now, only 2D version
template <const int n_ftrs>
class AdaptivePwMultiLinearMap2D : public GenAugMap<n_ftrs>
{
  	protected:

		// Basic properties of the adaptive map/mesh
		std::string mesh_file;
		std::array<int, n_ftrs> n_cells_per_dim_l0;
		std::array<double, n_ftrs> h_l0;
		int n_refinements;

		std::vector<std::vector<int>> mesh_tree;
		std::vector<bool> has_hanging_node;
		
		int n_leaves;
		// Note that n-dim hypercube has 3^n total 0-d, 1-d, ..., n-d entities
		// This includes corners, edges, faces, ..., up to the full n-d cell itself
		// Note that there are exactly 1 cell and 2^n corners per n-d hypercube
		const int n_possible_params_per_cell = pow(3, n_ftrs) - 1; // Excluding the full n-d cell itself
		std::vector<std::vector<int>> cell_to_param;

		// Stuff for the actual parameters
		std::array<double, n_ftrs>	min_ftrs;
		std::array<double, n_ftrs>	max_ftrs;
		// Note that parameter and dJ/dparameter vectors are inherited from GenAugMap

		// Other stuff
		int p_idw = 2;		// Power parameter for inverse distance weighting

 	public:
		AdaptivePwMultiLinearMap2D() : GenAugMap<n_ftrs>() { }

		inline void init(
			const std::string& name_,
			const std::string& mesh_file_,
			const std::array<double, n_ftrs>& min_ftrs_,
			const std::array<double, n_ftrs>& max_ftrs_
		)
		{
			this->name = name_;
			mesh_file = mesh_file_;
			min_ftrs = min_ftrs_;
			max_ftrs = max_ftrs_;

			// Now read and process the mesh file
			int n_params=0;
			for (int i_proc=0; i_proc<mpi_size; i_proc++) {
				if (i_proc==mpi_rank) {
					FILE* fp = fopen(mesh_file.c_str(), "r");
					if (fp == NULL) {
						PRINT("Error: Could not open AdaptivePwMultiLinearMap2D mesh file: %s", mesh_file.c_str());
						throw(-1);
					}
					
					// First number in first line contains the number of features/dimensions
					int n_ftrs_in_file;
					fscanf(fp, "%d", &n_ftrs_in_file);
					assert(n_ftrs_in_file==n_ftrs && "Number of features/dimensions in mesh file does not match template parameter n_ftrs");

					// Next n_ftrs numbers in first line contain the number of cells per dimension at level 0
					for (int i_ftr=0; i_ftr<n_ftrs; i_ftr++)
						fscanf(fp, "%d", &n_cells_per_dim_l0[i_ftr]);
					
					// Next (and last) number in first line contains the number of refinement levels
					fscanf(fp, "%d", &n_refinements);

					// Next section contains the mesh tree structure for each refinement level
					mesh_tree.resize(n_refinements+1);
					int n_leaves_counter = 0;
					for (int i_level=0; i_level<=n_refinements; i_level++) {
						// Calculate the number of cells in the current level
						int n_cells_level = 1;
						for (int i_ftr=0; i_ftr<n_ftrs; i_ftr++)
							n_cells_level *= n_cells_per_dim_l0[i_ftr] * (1<<i_level);
						
						mesh_tree[i_level].resize(n_cells_level);

						// Read the tree structure for the current level
						int tmp;
						for (int i_cell=0; i_cell<n_cells_level; i_cell++) {
						// for (int i_cell=0; i_cell<1; i_cell++) {
							fscanf(fp, "%d", &tmp);
							mesh_tree[i_level][i_cell] = tmp;
							if (tmp >= 0) {
								n_leaves_counter++;	
								n_leaves = tmp+1;
							}
							else if (tmp == -1) ; // Do nothing as cell is refined and not a leaf
							else if (tmp == -2) ; // Do nothing as cell does not exist
							else assert(false && "Invalid cell value of < -2 in mesh file");
						}
					}
							
					// Double check that number of leaves matches expected number
					assert(n_leaves_counter == n_leaves && "Number of leaves in mesh file does not match final cell index in mesh");
					
					// Final section contains the mapping from cells to param indices
					cell_to_param.resize(n_leaves);
					has_hanging_node.resize(n_leaves, false);
					for (int i_leaf=0; i_leaf<n_leaves; i_leaf++) {
						cell_to_param[i_leaf].resize(n_possible_params_per_cell);
						for (int i_pp=0; i_pp<n_possible_params_per_cell; i_pp++) {
							int tmp;
							fscanf(fp, "%d", &tmp);
							cell_to_param[i_leaf][i_pp] = tmp;
							n_params = std::max(n_params, tmp+1);
							if (tmp >= 0 && i_pp >= (1<<n_ftrs)) { 
								// This cell has a hanging node
								has_hanging_node[i_leaf] = true;
							}						
							
							if (tmp < -1)
								assert(false && "Invalid param index in cell to param mapping in mesh file");
						}
					}	
					fclose(fp);			
				}
				MPI_Barrier(mpi_comm);
			}

			// Calculate h_l0
			for (int i_ftr=0; i_ftr<n_ftrs; i_ftr++)
				h_l0[i_ftr] = (max_ftrs[i_ftr] - min_ftrs[i_ftr]) / n_cells_per_dim_l0[i_ftr];

			// Resize params and dJ_dParams vectors
			this->params.resize(n_params);
			this->dJ_dParams.resize(n_params);
		}
		double calculate_value(
			const std::array<double, n_ftrs>& ftrs,
			double* dBeta_dFtrs = nullptr
		)
		{
			// First need to find which cell the input features belong to
			// We will search from coarsest to finest level until we reach the appropriate leaf cell
			int cell, i_level;
			std::array<double, n_ftrs> ftrs0, ftrs1;
			for (i_level=0; i_level<=n_refinements; i_level++) {
				// Calculate the cell indices in this level
				int n_cells_dim, cell_index_dim, cell_index_lvl=0, offset=1;
				double h_dim;
				std::array<int, n_ftrs> cell_inds;
				for (int i_ftr=0; i_ftr<n_ftrs; i_ftr++) {
					// First determine cell index in this dimension
					n_cells_dim = n_cells_per_dim_l0[i_ftr] * (1<<i_level);
					h_dim = h_l0[i_ftr] / (1<<i_level);
					cell_index_dim = (int)((ftrs[i_ftr]-min_ftrs[i_ftr]) / h_dim);
					
					// Convert to unrolled cell index for this level
					cell_index_lvl += offset * cell_index_dim;

					// Calculate min and max feature coords for this dimension
					ftrs0[i_ftr] = min_ftrs[i_ftr] + cell_index_dim * h_dim;
					ftrs1[i_ftr] = ftrs0[i_ftr] + h_dim;

					// Prepare offset for next dimension
					offset *= n_cells_dim;
				}

				// Check the mesh tree to see if this cell is a leaf or needs to be refined further
				cell = mesh_tree[i_level][cell_index_lvl];
				if (cell >= 0) {
					// This is a leaf cell, we are done
					break;
				}
				else if (cell == -1) {
					// This cell is refined, continue to next level
					continue;
				}
				else {
					cout << "Error: Invalid cell value " << cell << " encountered when searching for leaf cell in AdaptivePwMultiLinearMap2D" << endl;
					cout << "At level " << i_level << " for cell index " << cell_index_lvl << endl;
					cout << "Feature values: ";
					for (int i_ftr=0; i_ftr<n_ftrs; i_ftr++)
						cout << ftrs[i_ftr] << " ";
					cout << endl;
					assert(false && "Invalid cell value encountered when searching for leaf cell");
				}
			}

			// Once we determine the leaf cell and min/max ftr vals, we interpolate using bilinear
			// interp if no hanging nodes or inverse distance weighting if there are hanging nodes
			double beta;
			if (!has_hanging_node[cell]) {
			// if (true) {			// For now always do bilinear interp to see if IDW is causing problems
				// Bilinear interp
				// here only assuming 2D because I cba with the math for nD right now
				double w00, w10, w01, w11, vol;
				w00 = (ftrs1[0] - ftrs[0]) * (ftrs1[1] - ftrs[1]);
				w10 = (ftrs[0] - ftrs0[0]) * (ftrs1[1] - ftrs[1]);
				w01 = (ftrs1[0] - ftrs[0]) * (ftrs[1] - ftrs0[1]);
				w11 = (ftrs[0] - ftrs0[0]) * (ftrs[1] - ftrs0[1]);
				vol = (ftrs1[0] - ftrs0[0]) * (ftrs1[1] - ftrs0[1]);

				beta = (w00 * this->params[cell_to_param[cell][0]] +
						w10 * this->params[cell_to_param[cell][1]] +
						w01 * this->params[cell_to_param[cell][2]] +
						w11 * this->params[cell_to_param[cell][3]]) / vol;
			} 
			else {
				// IDW
				// here only assuming 2D so we don't have to generalize weights for all non-vertex nodes
				beta = 0.0;
				double weight_sum = 0.0;
				double dist;
				
				// 0-0 corner
				dist = sqrt(pow(ftrs[0]-ftrs0[0], 2) + pow(ftrs[1]-ftrs0[1], 2));
				weight_sum += 1.0 / pow(dist, p_idw);
				beta += this->params[cell_to_param[cell][0]] / pow(dist, p_idw);

				// 1-0 corner
				dist = sqrt(pow(ftrs[0]-ftrs1[0], 2) + pow(ftrs[1]-ftrs0[1], 2));
				weight_sum += 1.0 / pow(dist, p_idw);
				beta += this->params[cell_to_param[cell][1]] / pow(dist, p_idw);

				// 0-1 corner
				dist = sqrt(pow(ftrs[0]-ftrs0[0], 2) + pow(ftrs[1]-ftrs1[1], 2));
				weight_sum += 1.0 / pow(dist, p_idw);
				beta += this->params[cell_to_param[cell][2]] / pow(dist, p_idw);

				// 1-1 corner
				dist = sqrt(pow(ftrs[0]-ftrs1[0], 2) + pow(ftrs[1]-ftrs1[1], 2));
				weight_sum += 1.0 / pow(dist, p_idw);
				beta += this->params[cell_to_param[cell][3]] / pow(dist, p_idw);

				// Now consider hanging nodes if they exist
				// Node at (0.5, 0)
				if (cell_to_param[cell][4] >= 0) {
					dist = sqrt(pow(ftrs[0]-(ftrs0[0]+ftrs1[0])/2, 2) + pow(ftrs[1]-ftrs0[1], 2));
					weight_sum += 1.0 / pow(dist, p_idw);
					beta += this->params[cell_to_param[cell][4]] / pow(dist, p_idw);
				}

				// Node at (0, 0.5)
				if (cell_to_param[cell][5] >= 0) {
					dist = sqrt(pow(ftrs[0]-ftrs0[0], 2) + pow(ftrs[1]-(ftrs0[1]+ftrs1[1])/2, 2));
					weight_sum += 1.0 / pow(dist, p_idw);
					beta += this->params[cell_to_param[cell][5]] / pow(dist, p_idw);
				}

				// Node at (1, 0.5)
				if (cell_to_param[cell][6] >= 0) {
					dist = sqrt(pow(ftrs[0]-ftrs1[0], 2) + pow(ftrs[1]-(ftrs0[1]+ftrs1[1])/2, 2));
					weight_sum += 1.0 / pow(dist, p_idw);
					beta += this->params[cell_to_param[cell][6]] / pow(dist, p_idw);
				}

				// Node at (0.5, 1)
				if (cell_to_param[cell][7] >= 0) {
					dist = sqrt(pow(ftrs[0]-(ftrs0[0]+ftrs1[0])/2, 2) + pow(ftrs[1]-ftrs1[1], 2));
					weight_sum += 1.0 / pow(dist, p_idw);
					beta += this->params[cell_to_param[cell][7]] / pow(dist, p_idw);
				}

				beta /= weight_sum;						
			}
			return beta;
		}

		void calculate_sens(
			const std::array<double, n_ftrs>& ftrs,
			const double& dJ_dBeta
		)
		{
			// First need to find which cell the input features belong to
			// We will search from coarsest to finest level until we reach the appropriate leaf cell
			int cell, i_level;
			std::array<double, n_ftrs> ftrs0, ftrs1;
			for (i_level=0; i_level<=n_refinements; i_level++) {
				// Calculate the cell indices in this level
				int n_cells_dim, cell_index_dim, cell_index_lvl=0, offset=1;
				double h_dim;
				std::array<int, n_ftrs> cell_inds;
				for (int i_ftr=0; i_ftr<n_ftrs; i_ftr++) {
					// First determine cell index in this dimension
					n_cells_dim = n_cells_per_dim_l0[i_ftr] * (1<<i_level);
					h_dim = h_l0[i_ftr] / (1<<i_level);
					cell_index_dim = (int)((ftrs[i_ftr]-min_ftrs[i_ftr]) / h_dim);
					
					// Convert to unrolled cell index for this level
					cell_index_lvl += offset * cell_index_dim;

					// Calculate min and max feature coords for this dimension
					ftrs0[i_ftr] = min_ftrs[i_ftr] + cell_index_dim * h_dim;
					ftrs1[i_ftr] = ftrs0[i_ftr] + h_dim;

					// Prepare offset for next dimension
					offset *= n_cells_dim;
				}

				// Check the mesh tree to see if this cell is a leaf or needs to be refined further
				cell = mesh_tree[i_level][cell_index_lvl];
				if (cell >= 0) {
					// This is a leaf cell, we are done
					break;
				}
				else if (cell == -1) {
					// This cell is refined, continue to next level
					continue;
				}
				else {
					assert(false && "Invalid cell value encountered when searching for leaf cell");
				}
			}

			// Once we determine the leaf cell, we calculate sensitivities
			if (!has_hanging_node[cell]) {
			// if (true) {			// For now always do bilinear interp to see if IDW is causing problems
				// Bilinear interp
				// here only assuming 2D because I cba with the math for nD right now
				double w00, w10, w01, w11, vol;
				w00 = (ftrs1[0] - ftrs[0]) * (ftrs1[1] - ftrs[1]);
				w10 = (ftrs[0] - ftrs0[0]) * (ftrs1[1] - ftrs[1]);
				w01 = (ftrs1[0] - ftrs[0]) * (ftrs[1] - ftrs0[1]);
				w11 = (ftrs[0] - ftrs0[0]) * (ftrs[1] - ftrs0[1]);
				vol = (ftrs1[0] - ftrs0[0]) * (ftrs1[1] - ftrs0[1]);

				this->dJ_dParams[cell_to_param[cell][0]] += dJ_dBeta * w00 / vol;
				this->dJ_dParams[cell_to_param[cell][1]] += dJ_dBeta * w10 / vol;
				this->dJ_dParams[cell_to_param[cell][2]] += dJ_dBeta * w01 / vol;
				this->dJ_dParams[cell_to_param[cell][3]] += dJ_dBeta * w11 / vol;
			} 
			else {
				// IDW
				// here only assuming 2D so we don't have to generalize weights for all non-vertex nodes
				double weight_sum = 0.0;
				std::vector<double> dist(n_possible_params_per_cell, 0.0);
				
				// 0-0 corner
				dist[0] = sqrt(pow(ftrs[0]-ftrs0[0], 2) + pow(ftrs[1]-ftrs0[1], 2));
				weight_sum += 1.0 / pow(dist[0], p_idw);

				// 1-0 corner
				dist[1] = sqrt(pow(ftrs[0]-ftrs1[0], 2) + pow(ftrs[1]-ftrs0[1], 2));
				weight_sum += 1.0 / pow(dist[1], p_idw);

				// 0-1 corner
				dist[2] = sqrt(pow(ftrs[0]-ftrs0[0], 2) + pow(ftrs[1]-ftrs1[1], 2));
				weight_sum += 1.0 / pow(dist[2], p_idw);

				// 1-1 corner
				dist[3] = sqrt(pow(ftrs[0]-ftrs1[0], 2) + pow(ftrs[1]-ftrs1[1], 2));
				weight_sum += 1.0 / pow(dist[3], p_idw);

				// Now consider hanging nodes if they exist
				// Node at (0.5, 0)
				if (cell_to_param[cell][4] >= 0) {
					dist[4] = sqrt(pow(ftrs[0]-(ftrs0[0]+ftrs1[0])/2, 2) + pow(ftrs[1]-ftrs0[1], 2));
					weight_sum += 1.0 / pow(dist[4], p_idw);
				}

				// Node at (0, 0.5)
				if (cell_to_param[cell][5] >= 0) {
					dist[5] = sqrt(pow(ftrs[0]-ftrs0[0], 2) + pow(ftrs[1]-(ftrs0[1]+ftrs1[1])/2, 2));
					weight_sum += 1.0 / pow(dist[5], p_idw);
				}

				// Node at (1, 0.5)
				if (cell_to_param[cell][6] >= 0) {
					dist[6] = sqrt(pow(ftrs[0]-ftrs1[0], 2) + pow(ftrs[1]-(ftrs0[1]+ftrs1[1])/2, 2));
					weight_sum += 1.0 / pow(dist[6], p_idw);
				}

				// Node at (0.5, 1)
				if (cell_to_param[cell][7] >= 0) {
					dist[7] = sqrt(pow(ftrs[0]-(ftrs0[0]+ftrs1[0])/2, 2) + pow(ftrs[1]-ftrs1[1], 2));
					weight_sum += 1.0 / pow(dist[7], p_idw);
				}

				// Now that we have all distances and weight sum, calculate sensitivities
				for (int i_pp=0; i_pp<n_possible_params_per_cell; i_pp++) {
					if (cell_to_param[cell][i_pp] >= 0) {
						this->dJ_dParams[cell_to_param[cell][i_pp]] += dJ_dBeta * (1.0 / pow(dist[i_pp], p_idw)) / weight_sum;
					}
				}
			}
		}
};




#endif