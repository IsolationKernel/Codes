# IDKC-Trajectory
The new IDK-based clustering algorithm, called IDKC, makes full use of the distributional kernel for trajectory similarity measuring and clustering. IDKC identifies non-linearly separable clusters with irregular shapes and varied densities in linear time.

## Requirements
- Python >= 3.5
- Matlab >= R2019a

## Datasets
All datasets are stored in `./datasets` as .mat files, containing trajectory data and labels.

## Similarity measure & trajectory representation

You can use IDK to generate vector embeddings of trajectories. Run `traj_embedding.py` under current directory:

```
python ./idk/traj_embedding.py
```
## Visualization with MDS
The embedding data is stored in `./embeddings`. You can also use MDS to visualize the embedding result:

```
python ./utils/trajMDS.py
```
## Trajectory clustering with IDKC

After generating the embedding of trajectories, run `./idkc/IDKC_traj.mlx` to do clustering.
