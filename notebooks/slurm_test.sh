#! /bin/bash


#SBATCH --job-name=CO_West_Central -t 12:00:00 -c10 --mem=190G

cd /panfs/ccds02/nobackup/people/sbhusha1/sw/lidar_tools/ 
pixi shell -e dev

# Change to job directory
cd /panfs/ccds02/nobackup/people/sbhusha1/pcd/Colorado/ 
time pdal_pipeline create-dsm processing_extent.geojson UTM_13N_WGS84_G2139_3D.wkt CO_PCD/CO_West_Central_2019_PCD 


