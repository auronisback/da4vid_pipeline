import os

DATA_PREPARATION_FOLDER = '/masif/data/masif_site/output/data_preparation'

masif_opts = {
  'raw_pdb_dir': os.path.join(DATA_PREPARATION_FOLDER, '00-raw_pdbs/'),
  'pdb_chain_dir': os.path.join(DATA_PREPARATION_FOLDER, '01-benchmark_pdbs/'),
  'ply_chain_dir': os.path.join(DATA_PREPARATION_FOLDER, '01-benchmark_surfaces/'),
  'ply_file_template': os.path.join(DATA_PREPARATION_FOLDER, '01-benchmark_surfaces', '{}_{}.ply'),
  'tmp_dir': os.path.join(DATA_PREPARATION_FOLDER, '.tmp'),

  'use_hbond': True,
  'use_hphob': True,
  'use_apbs': True,
  'compute_iface': True,

  'mesh_res': 1.0,
  'feature_interpolation': True,

  'radius': 12,

  'ppi_search': {},
  'site': {
    'training_list': 'lists/training.txt',
    'testing_list': 'lists/testing.txt',
    'masif_precomputation_dir': os.path.join(DATA_PREPARATION_FOLDER, '04a-precomputation_9A', 'precomputation/'),
    'model_dir': '/masif/data/masif_site/nn_models/all_feat_3l/model_data/',
    'n_conv_layers': 3,
    'range_val_samples': 0.9,
    'feat_mask': [1.0, 1.0, 1.0, 1.0, 1.0],
    'max_shape_size': 100,
    'max_distance': 9.0,
    'out_pred_dir': '/masif/data/masif_site/output/all_feat_3l/pred_data/',
    'out_surf_dir': '/masif/data/masif_site/output/all_feat_3l/pred_surfaces/',
  },
  'ligand': {}
}